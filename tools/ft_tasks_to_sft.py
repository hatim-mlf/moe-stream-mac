#!/usr/bin/env python3
"""Convert task folders into task-disjoint supervised fine-tuning JSONL files."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {"c": ".c", "python": ".py"}


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--valid-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261105)
    return parser.parse_args(argv)


def task_prompt(folder, meta):
    prompt = (folder / "spec.md").read_text().strip()
    language = meta["language"]
    fence = "c" if language == "c" else "python"
    if meta["kind"] == "skeleton":
        skeleton = (folder / f"skeleton{EXTENSIONS[language]}").read_text().strip()
        prompt += f"\n\n## Skeleton\n\n```{fence}\n{skeleton}\n```"
    if meta["kind"] == "review":
        attempt = (folder / meta["previous_attempt"]).read_text().strip()
        notes = "\n".join(f"- {note}" for note in meta["review_notes"])
        prompt += (f"\n\n## Previous attempt\n\n```{fence}\n{attempt}\n```"
                   f"\n\n## Review notes\n\n{notes}")
    return prompt


def load_dependencies():
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import ft_eval
    import ft_build_dataset
    if ft_eval.SYSTEM != ft_build_dataset.SYSTEM:
        raise RuntimeError("ft_eval.SYSTEM and ft_build_dataset.SYSTEM differ")
    return ft_eval, ft_build_dataset


def collect_rows(directory, ft_eval, ft_build_dataset):
    rows = {}
    for folder in sorted(path for path in directory.iterdir() if path.is_dir()):
        meta = json.loads((folder / "meta.json").read_text())
        task_id = meta["id"]
        if task_id != folder.name or task_id in rows:
            raise ValueError(f"invalid or duplicate task id: {task_id}")
        code = (folder / f"solution{EXTENSIONS[meta['language']]}").read_text()
        prompt = task_prompt(folder, meta)
        fence = "c" if meta["language"] == "c" else "python"
        row = ft_build_dataset.make_messages(prompt, ft_build_dataset.code_fence(code, fence))
        row["messages"][0]["content"] = ft_eval.SYSTEM
        rows[task_id] = row
    if not rows:
        raise ValueError(f"no task folders found in {directory}")
    return rows


def split_ids(task_ids, valid_frac, seed):
    if not 0.0 <= valid_frac < 1.0:
        raise ValueError("--valid-frac must be at least 0 and less than 1")
    shuffled = sorted(task_ids)
    random.Random(seed).shuffle(shuffled)
    valid_count = int(round(len(shuffled) * valid_frac))
    if valid_frac > 0 and len(shuffled) > 1:
        valid_count = max(1, valid_count)
    valid_ids = sorted(shuffled[:valid_count])
    train_ids = sorted(shuffled[valid_count:])
    return train_ids, valid_ids


def jsonl_bytes(rows):
    return "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                   for row in rows).encode("utf-8")


def write_dataset(out, rows, valid_frac, seed):
    train_ids, valid_ids = split_ids(list(rows), valid_frac, seed)
    train_data = jsonl_bytes([rows[task_id] for task_id in train_ids])
    valid_data = jsonl_bytes([rows[task_id] for task_id in valid_ids])
    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_bytes(train_data)
    (out / "valid.jsonl").write_bytes(valid_data)
    manifest = {"task_ids": sorted(rows), "train_task_ids": train_ids, "valid_task_ids": valid_ids,
                "valid_frac": valid_frac, "seed": seed,
                "sha256": {"train.jsonl": hashlib.sha256(train_data).hexdigest(),
                           "valid.jsonl": hashlib.sha256(valid_data).hexdigest()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_leak_check():
    result = subprocess.run([sys.executable, str(ROOT / "tools/ft_leak_check.py")], cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, end="")
    if result.returncode:
        raise RuntimeError("ft_leak_check.py failed")


def main(argv=None):
    args = arguments(argv)
    directory = args.directory.resolve()
    out = args.out if args.out.is_absolute() else ROOT / args.out
    try:
        ft_eval, ft_build_dataset = load_dependencies()
        rows = collect_rows(directory, ft_eval, ft_build_dataset)
        manifest = write_dataset(out, rows, args.valid_frac, args.seed)
        run_leak_check()
    except (OSError, ValueError, json.JSONDecodeError, RuntimeError) as error:
        print(f"dataset conversion failed: {error}", file=sys.stderr)
        return 1
    print(f"wrote {len(manifest['train_task_ids'])} train and {len(manifest['valid_task_ids'])} valid rows to {out}")
    print(json.dumps({"sha256": manifest["sha256"], "train_ids": len(manifest["train_task_ids"]),
                      "valid_ids": len(manifest["valid_task_ids"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
