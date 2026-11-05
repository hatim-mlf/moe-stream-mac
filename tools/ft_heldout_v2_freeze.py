#!/usr/bin/env python3
"""Generate the immutable source manifest for the heldout-v2 coding suite."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "evals" / "heldout-v2"
MANIFEST = ROOT / "tools" / "ft_eval_heldout_v2.json"
REQUIRED_TASK_KEYS = {
    "id", "lang", "target", "build_cmd", "test_cmd", "timeout_s", "classes", "kind", "difficulty"
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_names(task_dir, task):
    source = (task_dir / "reference" / task["target"]).read_text()
    if task["lang"] == "python":
        tree = ast.parse(source)
        names = set()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
                names.update(child.name for child in node.body
                             if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and not child.name.startswith("_"))
        return sorted(names)
    names = set()
    pattern = re.compile(r"(?m)^(?!static\b)\s*(?:int|void|float|double|size_t|uint8_t|int16_t|int32_t|TraceRing)\s+([A-Za-z_]\w*)\s*\(")
    names.update(pattern.findall(source))
    return sorted(names)


def collect_manifest(root=ROOT):
    suite = root / "evals" / "heldout-v2"
    entries = []
    digest_rows = []
    dirs = sorted(path for path in suite.iterdir() if path.is_dir())
    for task_dir in dirs:
        task_path = task_dir / "task.json"
        task = json.loads(task_path.read_text())
        if set(task) != REQUIRED_TASK_KEYS:
            raise ValueError(f"{task_path}: task.json keys do not match required schema")
        if task["id"] != task_dir.name or task["lang"] not in {"c", "python"}:
            raise ValueError(f"{task_path}: id or language mismatch")
        if task["kind"] not in {"kernel", "utility", "skeleton"}:
            raise ValueError(f"{task_path}: invalid task kind")
        if task["difficulty"] not in {"easy", "medium", "hard"}:
            raise ValueError(f"{task_path}: invalid difficulty")
        if not 1 <= task["timeout_s"] <= 60 or not task["test_cmd"]:
            raise ValueError(f"{task_path}: invalid timeout or missing test command")
        if task["lang"] == "c" and not task["build_cmd"]:
            raise ValueError(f"{task_path}: C task needs build_cmd")
        if task["lang"] == "python" and task["build_cmd"] is not None:
            raise ValueError(f"{task_path}: Python task build_cmd must be null")
        files = {"spec.md", "task.json"}
        if (task_dir / "review.md").is_file():
            files.add("review.md")
        for folder in ("tests", "fixtures"):
            path = task_dir / folder
            if path.is_dir():
                files.update(p.relative_to(task_dir).as_posix() for p in path.rglob("*") if p.is_file())
        hashes = {}
        for rel in sorted(files):
            path = task_dir / rel
            if not path.is_file():
                raise ValueError(f"{task_dir.name}: manifest input missing: {rel}")
            value = sha256(path)
            hashes[rel] = value
            digest_rows.append(f"{task_dir.name}/{rel}:{value}\n")
        if not (task_dir / "reference" / task["target"]).is_file():
            raise ValueError(f"{task_dir.name}: reference target is missing")
        mutant_dirs = sorted(p for p in (task_dir / "mutants").iterdir() if p.is_dir())
        if len(mutant_dirs) < 2:
            raise ValueError(f"{task_dir.name}: needs at least two mutants")
        for mutant_dir in mutant_dirs:
            if not (mutant_dir / task["target"]).is_file():
                raise ValueError(f"{task_dir.name}: mutant {mutant_dir.name} lacks target")
        entries.append({"id": task["id"], "sha256": hashes, "public_names": public_names(task_dir, task)})
    if len(entries) != 30:
        raise ValueError(f"expected 30 task directories, found {len(entries)}")
    review_count = sum((suite / task["id"] / "review.md").is_file() for task in entries)
    if review_count != 10:
        raise ValueError(f"expected 10 review prompts, found {review_count}")
    suite_hash = hashlib.sha256("".join(sorted(digest_rows)).encode()).hexdigest()
    return {"suite": "heldout-v2", "suite_sha256": suite_hash, "tasks": entries}


def main():
    payload = collect_manifest()
    MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"heldout-v2 frozen: {len(payload['tasks'])} tasks, suite_sha256={payload['suite_sha256']}")


if __name__ == "__main__":
    main()
