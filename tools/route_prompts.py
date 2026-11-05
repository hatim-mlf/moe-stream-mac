import argparse
import json
import os
import re
from pathlib import Path

import mlx.core as mx
import mlx_lm
import numpy as np

import mem_guard


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


def main(argv: list[str] | None = None) -> int:
    _total_memory = mem_guard.total_memory_bytes()
    _headroom = mem_guard.headroom_bytes_from_env(os.environ)
    print(mem_guard.startup_limit_message(_total_memory, _headroom), flush=True)
    mem_guard.start(total_bytes=_total_memory, headroom_bytes=_headroom)
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
        arr = np.array(ids, "<i4")
        arr.tofile(str(out_path))
        print(f"{item_id} {len(ids)} tokens")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
