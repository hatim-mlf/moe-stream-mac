# Task: write tools/route_prompts.py (Python 3.14, numpy + mlx_lm only)

Tokenize a JSON list of prompts into the C engine's prompt format, using the model's chat template, so we can record
routing traces under the same conditions as the evaluation.

Input JSON: a list of objects {"id": str, "prompt": str}. Output: one file <out_dir>/<id>.i32 per prompt, holding
the token ids as little-endian int32 (np.array(ids, "<i4").tofile(path)), nothing else.

Write exactly:

```python
def chat_ids(tokenizer, system: str | None, prompt: str) -> list[int]:
    """messages = [{"role": "system", "content": system}] if system else [], then the user message.
    Return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
    enable_thinking=False) as a list of int."""

def load_prompts(path: Path) -> list[dict]:
    """Read the JSON list; raise ValueError if it is not a list, if an item lacks a str "id" or str "prompt",
    if an id is not made of [A-Za-z0-9_-] only, or if ids repeat."""

def main(argv=None) -> int:
    """argparse: --model DIR (required), --prompts FILE (required), --out DIR (required, created if missing),
    --system-file FILE (optional; its text is the system message). Load the tokenizer with
    mlx_lm.utils.load_tokenizer(Path(model)). For each prompt write <out>/<id>.i32 and print "<id> <n> tokens".
    Return 0."""
```
End with `if __name__ == "__main__": raise SystemExit(main())`.
Rules: each function under 30 lines; no other functions. Return only the complete file in one fenced python block.


## Attempt 1 failed

Your code:
```
import argparse
import json
import re
from pathlib import Path

import mlx.core as mx
import mlx_lm


def chat_ids(tokenizer, system: str | None, prompt: str) -> list[int]:
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
    )


def load_prompts(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("JSON root is not a list")
    id_pattern = re.compile(r"^[A-Za-z0-9_-]+$")
    seen_ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Item is not a dict")
        item_id = item.get("id")
        item_prompt = item.get("prompt")
        if not isinstance(item_id, str) or not isinstance(item_prompt, str):
            raise ValueError("id and prompt must be strings")
        if not id_pattern.match(item_id):
            raise ValueError(f"Invalid id: {item_id}")
        if item_id in seen_ids:
            raise ValueError(f"Duplicate id: {item_id}")
        seen_ids.add(item_id)
    return data


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--prompts", required=True, type=str)
    parser.add_argument("--out", required=True, type=str)
    parser.add_argument("--system-file", type=str, default=None)
    args = parser.parse_args(argv)

    model_path = Path(args.model)
    prompts_path = Path(args.prompts)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = mlx_lm.utils.load_tokenizer(model_path)

    system_message = None
    if args.system_file is not None:
        sys_file = Path(args.system_file)
        if sys_file.exists():
            system_message = sys_file.read_text(encoding="utf-8").strip()

    prompts = load_prompts(prompts_path)

    for item in prompts:
        item_id = item["id"]
        item_prompt = item["prompt"]
        ids = chat_ids(tokenizer, system_message, item_prompt)
        out_path = out_dir / f"{item_id}.i32"
        arr = mx.array(ids, dtype=mx.int32)
        arr.tofile(str(out_path))
        print(f"{item_id} {len(ids)} tokens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Test output (last 80 lines):
```
ok  chat_ids
ok  rejects non-list
ok  rejects missing prompt
ok  rejects bad id
ok  rejects duplicate id
Traceback (most recent call last):
  File "/Users/hatim/moe-stream-mac/tools/test_route_prompts.py", line 57, in <module>
    main()
    ~~~~^^
  File "/Users/hatim/moe-stream-mac/tools/test_route_prompts.py", line 47, in main
    code = rp.main(["--model", str(MODEL), "--prompts", str(tmp / "p.json"), "--out", str(tmp / "out"),
                    "--system-file", str(tmp / "sys.txt")])
  File "/Users/hatim/moe-stream-mac/tools/route_prompts.py", line 71, in main
    arr.tofile(str(out_path))
    ^^^^^^^^^^
AttributeError: 'mlx.core.array' object has no attribute 'tofile'
```

Fix it. Apply every point; change nothing else.