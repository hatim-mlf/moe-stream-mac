# Task: write five definitions for engine/cli/qchat.py (Python 3.14, numpy)

Context (already in the file; do not repeat it): these imports and definitions exist.

```python
import sys, time
from dataclasses import dataclass, field
import numpy as np
DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
HELP = "...multi-line help text..."

@dataclass
class Settings:
    temp: float = 0.0; top_p: float = 0.95; top_k: int = 20; think: bool = False
    threads: int = 8; system: str = ""; max_tokens: int = 2048

@dataclass
class TurnStats:
    prompt_tokens: int = 0; prompt_s: float = 0.0; gen_tokens: int = 0; gen_s: float = 0.0
    extra: dict = field(default_factory=dict)      # extra["expert_gb"]: GB read from SSD this turn

# chat.engine.set_threads(n) -> int (effective count); chat.engine.stats() -> dict with keys
#   peak_rss_gb, expert_gb_read, expert_read_s, position, max_len;  chat.reset() starts a new conversation.
```

Write exactly these five definitions, with these signatures:

```python
def sample(logits: np.ndarray, s: Settings, rng: np.random.Generator) -> int:
    """temp <= 0 -> int(argmax). Otherwise: take the top_k largest logits (np.argpartition), compute
    softmax(logits / temp) over them in float64, sort descending, keep the smallest prefix whose cumulative
    probability >= top_p, renormalise, and return the chosen token id (int) drawn with rng.choice."""

class Printer:
    """Streams the reply. __init__(self, tokenizer, think_open: int, think_close: int).
    self.dt = tokenizer.detokenizer; self.dt.reset().
    feed(self, token: int): if token == think_open -> enter thinking mode (print DIM), return;
      if token == think_close -> leave thinking mode (print RESET), return; else self.dt.add_token(token) and
      print self.dt.last_segment with end="" and flush=True.
    finish(self): self.dt.finalize(); print self.dt.last_segment (end=""), then RESET and a newline."""

def footer(t: TurnStats, eng_stats: dict) -> str:
    """One DIM line like:
    [prompt 42 tok @ 3.1 tok/s | reply 120 tok @ 0.83 tok/s | ctx 162/4096 | RSS 5.5 GB | SSD 3.9 GB/tok]
    SSD GB/tok = t.extra["expert_gb"] / max(1, t.prompt_tokens + t.gen_tokens). Guard divisions by zero."""

def handle_command(line: str, s: Settings, chat) -> bool:
    """Return False for /exit (or /quit), True otherwise. Commands:
    /help -> print(HELP); /reset -> chat.reset(), print "context cleared";
    /threads N -> s.threads = chat.engine.set_threads(N), print it; /think on|off -> s.think;
    /temp X -> s.temp (float >= 0); /system TEXT -> s.system, print that it applies after /reset;
    /stats -> print chat.engine.stats() one "key: value" per line; anything else -> print(HELP).
    Bad or missing arguments: print a one-line error, keep the old value, never raise."""
```

Rules: no imports, no other code, each definition under 40 lines. Return only these five definitions in one fenced python block.


## Your previous attempt failed

Your sample and handle_command:
```python
def sample(logits: np.ndarray, s: Settings, rng: np.random.Generator) -> int:
    if s.temp <= 0:
        return int(np.argmax(logits))
    k = min(s.top_k, len(logits))
    indices = np.argpartition(logits, -k)[-k:]
    subset = logits[indices].astype(np.float64)
    probs = np.exp(subset / s.temp)
    probs /= probs.sum()
    sorted_idx = np.argsort(subset)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, rng.random() * cumsum[-1])
    chosen = indices[sorted_idx[cutoff]]
    return int(chosen)


def handle_command(line: str, s: Settings, chat) -> bool:
    parts = line.strip().split(maxsplit=1)
    cmd = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    try:
        if cmd in ("/exit", "/quit"):
            return False
        if cmd == "/help":
            print(HELP)
        elif cmd == "/reset":
            chat.reset()
            print("context cleared")
        elif cmd == "/threads":
            n = int(args)
            s.threads = chat.engine.set_threads(n)
            print(f"threads set to {s.threads}")
        elif cmd == "/think":
            s.think = args.lower() == "on"
        elif cmd == "/temp":
            val = float(args)
            if val >= 0:
                s.temp = val
        elif cmd == "/system":
            s.system = args
            print("system prompt set (applied after /reset)")
        elif cmd == "/stats":
            for k, v in chat.engine.stats().items():
                print(f"{k}: {v}")
        else:
            print(HELP)
    except Exception:
        pass
    return True


```

Code review: Printer and footer are accepted. Fix only these two and return ONLY sample and handle_command:
1. sample: top_p is never applied. After sorting the top_k probabilities descending, keep the smallest prefix whose
   cumulative sum >= s.top_p (at least 1 token), renormalise that prefix, then draw with rng.choice over it.
   Also subtract the max before exp: probs = np.exp((subset - subset.max()) / s.temp).
2. handle_command: never `except Exception: pass`. Validate each argument explicitly and print a one-line error
   ("usage: /threads N (N >= 1)" etc.) on bad input. /think must accept only "on" or "off" and print the new state;
   /temp must reject negatives with an error and print the new value on success.


Fix it. Reply with the complete corrected file.