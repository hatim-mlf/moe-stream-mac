```python
"""qchat: terminal chat with any Qwen3.5/3.6 MoE MLX checkpoint, running on the C engine (CPU, low memory).

  python engine/cli/qchat.py --model ~/models/Qwen3.5-122B-A10B-8bit --threads 8

The C engine (build/libqengine.dylib) does all model math; Python only handles the tokenizer, Qwen's chat
template, sampling and the terminal. Multi-turn chats only APPEND tokens (the linear-attention state cannot be
rewound), so earlier turns are never re-processed.
"""
import argparse
import ctypes
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from mlx_lm.utils import load_tokenizer

LIB = Path(__file__).resolve().parents[1] / "build" / "libqengine.dylib"
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
HELP = """commands:
  /help                 this help
  /reset                start a new conversation (clears context)
  /threads N            CPU threads for the engine
  /think on|off         let the model think before answering
  /temp X               sampling temperature (0 = greedy, deterministic)
  /system TEXT          set the system prompt (applies after /reset)
  /stats                memory, disk reads, context used
  /exit                 quit"""


class Engine:
    """ctypes binding to libqengine.dylib."""

    def __init__(self, model_dir: Path, ctx: int):
        self.lib = ctypes.CDLL(str(LIB))
        L = self.lib
        L.qe_load.restype = ctypes.c_void_p; L.qe_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        for name in ("qe_reset", "qe_free"):
            getattr(L, name).argtypes = [ctypes.c_void_p]
        L.qe_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        for name in ("qe_vocab", "qe_position", "qe_max_len"):
            getattr(L, name).argtypes = [ctypes.c_void_p]
        L.qe_set_threads.argtypes = [ctypes.c_int]
        for name in ("qe_expert_gb_read", "qe_expert_read_seconds", "qe_peak_rss_gb"):
            getattr(L, name).restype = ctypes.c_double
        self.handle = L.qe_load(str(model_dir).encode(), ctx)
        if not self.handle:
            raise SystemExit(f"engine could not load {model_dir} (see the message above)")
        self.vocab = L.qe_vocab(self.handle)
        self.logits = np.zeros(self.vocab, dtype=np.float32)
        self._ptr = self.logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    def forward(self, token: int) -> np.ndarray:
        rc = self.lib.qe_forward(self.handle, int(token), self._ptr)
        if rc == -2:
            raise ContextFull()
        if rc != 0:
            raise RuntimeError("engine forward failed")
        return self.logits

    def reset(self) -> None: self.lib.qe_reset(self.handle)
    def position(self) -> int: return self.lib.qe_position(self.handle)
    def max_len(self) -> int: return self.lib.qe_max_len(self.handle)
    def set_threads(self, n: int) -> int: return self.lib.qe_set_threads(n)
    def stats(self) -> dict:
        L = self.lib
        return {"peak_rss_gb": L.qe_peak_rss_gb(), "expert_gb_read": L.qe_expert_gb_read(),
                "expert_read_s": L.qe_expert_read_seconds(), "position": self.position(), "max_len": self.max_len()}


class ContextFull(Exception):
    pass


@dataclass
class Settings:
    temp: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    think: bool = False
    threads: int = 8
    system: str = ""
    max_tokens: int = 2048


@dataclass
class TurnStats:
    prompt_tokens: int = 0
    prompt_s: float = 0.0
    gen_tokens: int = 0
    gen_s: float = 0.0
    extra: dict = field(default_factory=dict)


def sample(logits: np.ndarray, s: Settings, rng: np.random.Generator) -> int:
    """TODO(qwen): temp <= 0 -> argmax. Otherwise: keep the top_k logits, softmax(logits / temp) in float64,
    keep the smallest set of those whose probabilities sum to >= top_p, renormalise, draw with rng."""
    if s.temp <= 0:
        return int(np.argmax(logits))
    logits = np.sort(logits)[::-1][:s.top_k]
    probs = np.exp(logits / s.temp).astype(np.float64)
    probs /= probs.sum()
    sorted_idx = np.argsort(logits)[::-1]
    cumsum = np.cumsum(probs)
    cutoff_idx = np.searchsorted(cumsum, s.top_p, side='right') + 1
    probs = probs[:cutoff_idx]
    sorted_idx = sorted_idx[:cutoff_idx]
    probs /= probs.sum()
    return int(sorted_idx[rng.choice(len(probs), p=probs)])


class Printer:
    """TODO(qwen): streams the reply. feed(token_id) receives every generated token in order.
    - Use the tokenizer's streaming detokenizer: self.dt = tokenizer.detokenizer; reset() in __init__;
      for each token call self.dt.add_token(token) and print self.dt.last_segment (no newline, flush).
    - Tokens with id think_open / think_close switch "thinking" mode on / off: text printed while thinking is
      wrapped in DIM ... RESET. Do not print the  tags themselves.
    - finish(): call self.dt.finalize(), print its last_segment, then RESET and a newline."""

    def __init__(self, tokenizer, think_open: int, think_close: int):
        self.dt = tokenizer.detokenizer
        self.dt.reset()
        self.think_open = think_open
        self.think_close = think_close
        self.in_think = False

    def feed(self, token: int) -> None:
        if token == self.think_open:
            self.in_think = True
            return
        if token == self.think_close:
            self.in_think = False
            return
        self.dt.add_token(token)
        seg = self.dt.last_segment
        if seg:
            if self.in_think:
                print(f"{DIM}{seg}{RESET}", end="", flush=True)
            else:
                print(seg, end="", flush=True)

    def finish(self) -> None:
        final = self.dt.finalize()
        if final:
            print(final, end="", flush=True)
        print(RESET + "\n", end="", flush=True)


def footer(t: TurnStats, eng_stats: dict) -> str:
    """TODO(qwen): one dim line, e.g.
    [prompt 42 tok @ 3.1 tok/s | reply 120 tok @ 0.83 tok/s | ctx 162/4096 | RSS 5.5 GB | SSD 3.9 GB/tok]
    SSD GB/tok = t.extra["expert_gb"] / max(1, t.prompt_tokens + t.gen_tokens). Guard every division by zero."""
    def safe_div(num, den):
        return num / max(1, den)
    
    p_tok = t.prompt_tokens
    p_s = t.prompt_s
    g_tok = t.gen_tokens
    g_s = t.gen_s
    pos = eng_stats["position"]
    mx = eng_stats["max_len"]
    rss = eng_stats["peak_rss_gb"]
    ssd = t.extra.get("expert_gb", 0.0)
    
    p_rate = safe_div(p_tok, p_s)
    g_rate = safe_div(g_tok, g_s)
    ssd_rate = safe_div(ssd, p_tok + g_tok)
    
    return (f"{DIM}[prompt {p_tok} tok @ {p_rate:.1f} tok/s | reply {g_tok} tok @ {g_rate:.2f} tok/s | "
            f"ctx {pos}/{mx} | RSS {rss:.2f} GB | SSD {ssd_rate:.2f} GB/tok]{RESET}")


def handle_command(line: str, s: Settings, chat: "Chat") -> bool:
    """TODO(qwen): handle the commands listed in HELP. Return False for /exit, True otherwise.
    /threads N -> s.threads = chat.engine.set_threads(N) and print it; /reset -> chat.reset();
    /think on|off -> s.think; /temp X -> s.temp (float >= 0); /system TEXT -> s.system and say it applies after /reset;
    /stats -> print chat.engine.stats() one key per line; unknown command -> print HELP.
    Bad arguments print a one-line error and keep the old value (never raise)."""
    parts = line.split()
    cmd = parts[0]
    
    if cmd == "/exit":
        return False
    elif cmd == "/help":
        print(HELP)
        return True
    elif cmd == "/reset":
        chat.reset()
        print("Context reset.")
        return True
    elif cmd == "/threads":
        if len(parts) < 2:
            print("Error: /threads requires N")
            return True
        try:
            n = int(parts[1])
            s.threads = chat.engine.set_threads(n)
            print(f"Threads set to {s.threads}")
        except ValueError:
            print("Error: N must be an integer")
        return True
    elif cmd == "/think":
        if len(parts) < 2:
            print("Error: /think requires on|off")
            return True
        val = parts[1].lower()
        if val == "on":
            s.think = True
            print("Thinking mode: ON")
        elif val == "off":
            s.think = False
            print("Thinking mode: OFF")
        else:
            print("Error: /think requires on|off")
        return True
    elif cmd == "/temp":
        if len(parts) < 2:
            print("Error: /temp requires X")
            return True
        try:
            val = float(parts[1])
            if val < 0:
                print("Error: temperature must be >= 0")
            else:
                s.temp = val
                print(f"Temperature set to {s.temp}")
        except ValueError:
            print("Error: X must be a float")
        return True
    elif cmd == "/system":
        if len(parts) < 2:
            print("Error: /system requires TEXT")
            return True
        s.system = " ".join(parts[1:])
        print(f"System prompt set. It will apply after /reset.")
        return True
    elif cmd == "/stats":
        stats = chat.engine.stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
        return True
    else:
        print(HELP)
        return True


class Chat:
    """Append-only conversation over the engine (written by the lead)."""

    def __init__(self, engine: Engine, tokenizer, model_dir: Path, s: Settings):
        self.engine, self.tok, self.s = engine, tokenizer, s
        gen = json.loads((model_dir / "generation_config.json").read_text()) if (model_dir / "generation_config.json").exists() else {}
        eos = gen.get("eos_token_id", [])
        self.stop = set(eos if isinstance(eos, list) else [eos]) | set(getattr(tokenizer, "eos_token_ids", []) or [])
        self.think_open = tokenizer.convert_tokens_to_ids("")
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.engine.reset()
        self.turns = 0
        self.closed = True   # whether the last assistant reply ended with