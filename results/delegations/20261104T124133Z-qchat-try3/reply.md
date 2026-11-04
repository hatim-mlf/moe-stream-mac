```python
def sample(logits: np.ndarray, s: Settings, rng: np.random.Generator) -> int:
    if s.temp <= 0:
        return int(np.argmax(logits))
    k = min(s.top_k, len(logits))
    indices = np.argpartition(logits, -k)[-k:]
    subset = logits[indices].astype(np.float64)
    probs = np.exp((subset - subset.max()) / s.temp)
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
```