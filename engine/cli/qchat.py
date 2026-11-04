"""qchat: terminal chat with any Qwen3.5/3.6 MoE MLX checkpoint, running on the C engine (CPU, low memory).

  python engine/cli/qchat.py --model ~/models/Qwen3.5-122B-A10B-8bit --threads 8

The C engine (build/libqengine.dylib) does all model math; Python only handles the tokenizer, Qwen's chat
template, sampling and the terminal. Multi-turn chats only APPEND tokens (the linear-attention state cannot be
rewound), so earlier turns are never re-processed.
"""
import argparse
import ctypes
import json
import os
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
    threads: int = 6
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
    """Greedy when temp <= 0; otherwise top-k, then top-p (nucleus), then a weighted draw."""
    if s.temp <= 0:
        return int(np.argmax(logits))
    k = max(1, min(s.top_k, len(logits)))
    top = np.argpartition(logits, -k)[-k:]
    top = top[np.argsort(logits[top])[::-1]]                      # descending by logit
    z = logits[top].astype(np.float64) / s.temp
    probs = np.exp(z - z.max()); probs /= probs.sum()
    keep = int(np.searchsorted(np.cumsum(probs), s.top_p)) + 1    # smallest prefix with mass >= top_p
    probs = probs[:keep] / probs[:keep].sum()
    return int(top[rng.choice(keep, p=probs)])


class Printer:
    def __init__(self, tokenizer, think_open: int, think_close: int):
        self.tokenizer = tokenizer
        self.dt = tokenizer.detokenizer
        self.dt.reset()
        self.think_open = think_open
        self.think_close = think_close

    def feed(self, token: int):
        if token == self.think_open:
            print(DIM, end="", flush=True)
            return
        if token == self.think_close:
            print(RESET, end="", flush=True)
            return
        self.dt.add_token(token)
        print(self.dt.last_segment, end="", flush=True)

    def finish(self):
        self.dt.finalize()
        print(self.dt.last_segment, end="")
        print(RESET)


def footer(t: TurnStats, eng_stats: dict) -> str:
    total_tokens = max(1, t.prompt_tokens + t.gen_tokens)
    prompt_rate = t.prompt_tokens / max(1e-9, t.prompt_s)
    gen_rate = t.gen_tokens / max(1e-9, t.gen_s)
    expert_gb = t.extra.get("expert_gb", 0.0)
    expert_per_tok = expert_gb / total_tokens
    pos = eng_stats.get("position", 0)
    max_len = eng_stats.get("max_len", 0)
    rss = eng_stats.get("peak_rss_gb", 0.0)
    return (f"{DIM}[prompt {t.prompt_tokens} tok @ {prompt_rate:.1f} tok/s "
            f"| reply {t.gen_tokens} tok @ {gen_rate:.2f} tok/s "
            f"| ctx {pos}/{max_len} | RSS {rss:.1f} GB "
            f"| experts read {expert_per_tok:.1f} GB/tok]{RESET}")


def handle_command(line: str, s: Settings, chat) -> bool:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    if cmd in ("/exit", "/quit"):
        return False
    if cmd == "/help":
        print(HELP)
    elif cmd == "/reset":
        chat.reset()
        print("context cleared")
    elif cmd == "/threads":
        try:
            n = int(args)
            if n < 1:
                print("usage: /threads N (N >= 1)")
            else:
                s.threads = chat.engine.set_threads(n)
                print(f"threads set to {s.threads}")
        except ValueError:
            print("usage: /threads N (N >= 1)")
    elif cmd == "/think":
        if args.lower() == "on":
            s.think = True
            print("think mode on")
        elif args.lower() == "off":
            s.think = False
            print("think mode off")
        else:
            print("usage: /think on|off")
    elif cmd == "/temp":
        try:
            val = float(args)
            if val < 0:
                print("usage: /temp X (X >= 0)")
            else:
                s.temp = val
                print(f"temp set to {s.temp}")
        except ValueError:
            print("usage: /temp X (X >= 0)")
    elif cmd == "/system":
        s.system = args
        print("system prompt set (applied after /reset)")
    elif cmd == "/stats":
        for k, v in chat.engine.stats().items():
            print(f"{k}: {v}")
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
        self.think_open = tokenizer.convert_tokens_to_ids("<think>")
        self.think_close = tokenizer.convert_tokens_to_ids("</think>")
        self.rng = np.random.default_rng()
        self.reset()

    def reset(self) -> None:
        self.engine.reset()
        self.turns = 0
        self.closed = True   # whether the last assistant reply ended with <|im_end|> fed into the engine

    def _render_turn(self, user: str) -> list[int]:
        msgs = ([{"role": "system", "content": self.s.system}] if self.s.system and self.turns == 0 else [])
        msgs.append({"role": "user", "content": user})
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=self.s.think)
        if self.turns > 0:
            text = ("" if self.closed else "<|im_end|>") + "\n" + text
        return self.tok.encode(text)

    def ask(self, user: str) -> TurnStats:
        t = TurnStats()
        ids = self._render_turn(user)
        gb0 = self.engine.stats()["expert_gb_read"]
        start = time.perf_counter()
        logits = None
        for tok in ids:
            logits = self.engine.forward(tok)
        t.prompt_tokens, t.prompt_s = len(ids), time.perf_counter() - start
        printer = Printer(self.tok, self.think_open, self.think_close)
        start = time.perf_counter()
        self.closed = False
        while t.gen_tokens < self.s.max_tokens:
            tok = sample(logits, self.s, self.rng)
            t.gen_tokens += 1
            if tok in self.stop:
                self.closed = False   # the stop token itself is not fed; the next turn feeds "<|im_end|>\n"
                break
            printer.feed(tok)
            logits = self.engine.forward(tok)
        printer.finish()
        t.gen_s = time.perf_counter() - start
        t.extra["expert_gb"] = self.engine.stats()["expert_gb_read"] - gb0
        self.turns += 1
        return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="MLX qwen3_5_moe checkpoint directory")
    ap.add_argument("--threads", type=int, default=6,
                    help="CPU threads (default 6 of 10: ~84%% of max speed, leaves cores for your other apps)")
    ap.add_argument("--nice", type=int, default=10, help="scheduling priority, 0-20 (default 10: other apps go first)")
    ap.add_argument("--ctx", type=int, default=4096, help="context length in tokens")
    ap.add_argument("--temp", type=float, default=0.0, help="0 = greedy (default, deterministic)")
    ap.add_argument("--think", action="store_true", help="enable thinking mode")
    ap.add_argument("--system", default="")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--prompt", help="answer one prompt and exit (non-interactive)")
    a = ap.parse_args()
    model_dir = a.model.expanduser()
    if a.nice:
        os.nice(max(0, min(20, a.nice)))
    if not LIB.exists():
        raise SystemExit(f"{LIB} missing: run `make -C engine lib`")
    s = Settings(temp=a.temp, think=a.think, threads=a.threads, system=a.system, max_tokens=a.max_tokens)
    t0 = time.perf_counter()
    engine = Engine(model_dir, a.ctx)
    s.threads = engine.set_threads(a.threads)
    chat = Chat(engine, load_tokenizer(model_dir), model_dir, s)
    print(f"{BOLD}{model_dir.name}{RESET} loaded in {time.perf_counter() - t0:.1f}s | {s.threads} threads | "
          f"ctx {a.ctx} | nice {a.nice} | RSS {engine.stats()['peak_rss_gb']:.2f} GB | /help for commands")
    prompts = [a.prompt] if a.prompt else None
    while True:
        try:
            line = prompts.pop(0) if prompts is not None else input(f"{BOLD}> {RESET}")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line.strip():
            if prompts is not None and not prompts: break
            continue
        if line.startswith("/"):
            if not handle_command(line.strip(), s, chat): break
            continue
        try:
            stats = chat.ask(line)
            print(footer(stats, engine.stats()))
        except ContextFull:
            print(f"\ncontext full ({engine.max_len()} tokens): /reset to start over")
        except KeyboardInterrupt:
            print("\n[interrupted: /reset recommended]")
        if prompts is not None and not prompts:
            break


if __name__ == "__main__":
    main()
