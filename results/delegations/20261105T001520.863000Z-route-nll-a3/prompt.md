# Task: write tools/route_nll.py (Python 3.14, numpy + mlx + mlx_lm only)

Measure how likely the model finds known-good answers (teacher-forced negative log-likelihood), with or without a
router bias from tools/router_bias.py. Used as the development metric for the router-bias sweep.

Input: a JSONL file; each line {"task_id": str, "messages": [..., {"role": "assistant", "content": str}]} where the
LAST message is the assistant answer to score.

Write exactly:

```python
def split_ids(tokenizer, messages: list) -> tuple[list[int], list[int]]:
    """prompt = tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, tokenize=True,
    enable_thinking=False); full = tokenizer.apply_chat_template(messages, tokenize=True, enable_thinking=False).
    Raise ValueError if the last message is not role "assistant" or full[:len(prompt)] != prompt.
    Return (prompt, full) as lists of int."""

def completion_nll(model, prompt: list[int], full: list[int]) -> tuple[float, int]:
    """logits = model(mx.array([full[:-1]])) -> shape [1, len(full)-1, vocab]. Convert to float32 and take
    log-softmax over the vocab (logits - mx.logsumexp(logits, axis=-1, keepdims=True)). The answer tokens are
    full[len(prompt):]; token full[j] is predicted at position j-1. Return (sum of -log p over the answer tokens as a
    Python float, number of answer tokens)."""

def score_file(model, tokenizer, path: Path) -> dict:
    """For every line: (prompt, full) = split_ids(...); nll, n = completion_nll(...). Return
    {"rows": [{"task_id", "nll", "tokens", "nll_per_token"}...], "total_nll", "total_tokens",
    "nll_per_token": total_nll / total_tokens}."""

def main(argv=None) -> int:
    """argparse: --model DIR (required), --data JSONL (required, repeatable: action="append"), --bias NPY (optional),
    --mode (choices router_bias.MODES, default "select"), --scale FLOAT (default 1.0; the bias array is multiplied by
    it), --out JSON (required). model, tokenizer = mlx_lm.load(model). If --bias: router_bias.install_router_bias(
    model, np.load(bias) * scale, mode). Write {"model", "bias", "mode", "scale", "files": {data path: score_file
    result}} to --out (indent=1) and print one line per file: "<path> nll/token=<value:.4f> tokens=<n>". Return 0."""
```
Import router_bias with `sys.path.insert(0, str(Path(__file__).resolve().parent))` then `import router_bias`.
End with `if __name__ == "__main__": raise SystemExit(main())`.
Rules: each function under 30 lines; no other functions. Return only the complete file in one fenced python block.


## Attempt 2 failed

Your code:
```
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
    answer_tokens = full[len(prompt):]
    if not answer_tokens:
        return (0.0, 0)
    logits = model(mx.array([full[:-1]]))
    logits = logits.astype(mx.float32)
    log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    answer_mx = mx.array(answer_tokens)
    nlls = -mx.take_along_axis(log_probs, mx.expand_dims(answer_mx, -1), axis=-1).squeeze(-1)
    return (float(nlls.sum()), len(answer_tokens))


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

Test output (last 80 lines):
```
ok  split_ids
Traceback (most recent call last):
  File "/Users/hatim/moe-stream-mac/tools/test_route_nll.py", line 65, in <module>
    main()
    ~~~~^^
  File "/Users/hatim/moe-stream-mac/tools/test_route_nll.py", line 46, in main
    nll, n = rn.completion_nll(uniform_model(vocab), small_full[:2], small_full)
             ~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/hatim/moe-stream-mac/tools/route_nll.py", line 32, in completion_nll
    nlls = -mx.take_along_axis(log_probs, mx.expand_dims(answer_mx, -1), axis=-1).squeeze(-1)
            ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
ValueError: [take_along_axis] Indices of dimension 2 does not match array of dimension 3.
```

Fix it. Apply every point; change nothing else.