```python
import argparse
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_bias


def split_ids(tokenizer, messages: list) -> tuple[list[int], list[int]]:
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("Last message must be role 'assistant'")
    prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True, enable_thinking=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False)
    if not full[:len(prompt)] == prompt:
        raise ValueError("Full does not start with prompt")
    return (prompt, full)


def completion_nll(model, prompt: list[int], full: list[int]) -> tuple[float, int]:
    lp = log_probs[0, len(prompt) - 1:, :]                    # [n_answer, vocab]
    answer = mx.array(full[len(prompt):])                      # [n_answer]
    picked = mx.take_along_axis(lp, answer[:, None], axis=-1)[:, 0]
    return (float(-picked.sum().item()), len(full) - len(prompt))


def score_file(model, tokenizer, path: Path) -> dict:
    rows = []
    total_nll = 0.0
    total_tokens = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            task_id = entry["task_id"]
            messages = entry["messages"]
            prompt, full = split_ids(tokenizer, messages)
            nll, n = completion_nll(model, prompt, full)
            nll_per_token = nll / n if n > 0 else 0.0
            rows.append({"task_id": task_id, "nll": nll, "tokens": n, "nll_per_token": nll_per_token})
            total_nll += nll
            total_tokens += n
    nll_per_token = total_nll / total_tokens if total_tokens > 0 else 0.0
    return {"rows": rows, "total_nll": total_nll, "total_tokens": total_tokens, "nll_per_token": nll_per_token}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, action="append")
    parser.add_argument("--bias", type=str, default=None)
    parser.add_argument("--mode", choices=router_bias.MODES, default="select")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    model, tokenizer = mlx_lm.load(args.model)
    if args.bias:
        router_bias.install_router_bias(model, np.load(args.bias) * args.scale, args.mode)
    files = {}
    for data_path in args.data:
        p = Path(data_path)
        files[str(p)] = score_file(model, tokenizer, p)
    output = {
        "model": args.model,
        "bias": args.bias,
        "mode": args.mode,
        "scale": args.scale,
        "files": files
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=1)
    for data_path in args.data:
        p = Path(data_path)
        result = files[str(p)]
        npt = result["nll_per_token"]
        n = result["total_tokens"]
        print(f"{p} nll/token={npt:.4f} tokens={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```