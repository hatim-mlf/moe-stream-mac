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
            f"| SSD {expert_per_tok:.1f} GB/tok]{RESET}")

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