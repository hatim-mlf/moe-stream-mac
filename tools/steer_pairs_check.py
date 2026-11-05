#!/usr/bin/env python3
"""Run every residual-steering contrast against its task-local test."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ft_eval

PAIRS_PATH = ROOT / "data/steering/pairs.jsonl"
TESTS_ROOT = ROOT / "data/steering/tests"
FENCE = re.compile(r"\A```(python|py|c)\s*\n(.*?)\n```\Z", re.DOTALL)


def fail(message):
    print(message, file=sys.stderr)


def main():
    if not PAIRS_PATH.is_file():
        fail(f"missing contrast set: {PAIRS_PATH}")
        return 1
    pairs = []
    with PAIRS_PATH.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                fail(f"{PAIRS_PATH}:{line_number}: invalid JSON: {error}")
                return 1
            pairs.append(row)

    errors = []
    required = {"pair_id", "task_id", "category", "split", "messages", "preferred", "unpreferred"}
    ids = set()
    task_ids = set()
    category_counts = Counter()
    split_counts = Counter()
    language_counts = Counter()
    for index, pair in enumerate(pairs, 1):
        prefix = f"row {index}"
        missing = required - set(pair)
        if missing:
            errors.append(f"{prefix}: missing fields {sorted(missing)}")
            continue
        if pair["pair_id"] in ids:
            errors.append(f"{prefix}: duplicate pair_id {pair['pair_id']}")
        ids.add(pair["pair_id"])
        if pair["task_id"] in task_ids:
            errors.append(f"{prefix}: duplicate task_id {pair['task_id']}")
        task_ids.add(pair["task_id"])
        if pair["category"] not in {"spec", "review", "api"}:
            errors.append(f"{prefix}: invalid category {pair['category']!r}")
        if pair["split"] not in {"build", "dev"}:
            errors.append(f"{prefix}: invalid split {pair['split']!r}")
        category_counts[pair["category"]] += 1
        split_counts[pair["split"]] += 1
        if not isinstance(pair["messages"], list) or len(pair["messages"]) != 2:
            errors.append(f"{prefix}: messages must contain exactly system and user messages")
        elif (pair["messages"][0] != {"role": "system", "content": ft_eval.SYSTEM} or
              pair["messages"][1].get("role") != "user"):
            errors.append(f"{prefix}: system prompt differs from ft_eval.SYSTEM or user role is invalid")
        answers = {}
        for name in ("preferred", "unpreferred"):
            answer = pair[name]
            match = FENCE.fullmatch(answer) if isinstance(answer, str) else None
            if not match:
                errors.append(f"{prefix}: {name} must be exactly one fenced code block")
                continue
            language, code = match.groups()
            answers[name] = ("python" if language in {"python", "py"} else "c", code)
        if "preferred" not in answers or "unpreferred" not in answers:
            continue
        if answers["preferred"][0] != answers["unpreferred"][0]:
            errors.append(f"{prefix}: answer languages differ")
            continue
        language = answers["preferred"][0]
        declared = pair.get("language", language)
        if declared not in {"python", "c99"} or ("python" if declared == "python" else "c") != language:
            errors.append(f"{prefix}: language metadata does not match code fences")
            continue
        language_counts[declared] += 1
        test_dir = TESTS_ROOT / pair["task_id"]
        test_file = test_dir / ("test.py" if language == "python" else "test.c")
        if not test_file.is_file():
            errors.append(f"{prefix}: missing task test {test_file.relative_to(ROOT)}")
            continue

        outcomes = {}
        for name in ("preferred", "unpreferred"):
            with tempfile.TemporaryDirectory(prefix="steer-pair-") as temp_name:
                temp = Path(temp_name)
                code_path = temp / ("candidate.py" if language == "python" else "candidate.c")
                code_path.write_text(answers[name][1] + "\n")
                if language == "python":
                    env = dict(os.environ)
                    env["PYTHONPATH"] = str(temp) + os.pathsep + env.get("PYTHONPATH", "")
                    proc = subprocess.run([sys.executable, str(test_file)], cwd=temp, env=env,
                                          text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                else:
                    compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
                    if not compiler:
                        errors.append("no C compiler found (need cc, gcc, or clang)")
                        return 1
                    binary = temp / "check"
                    compile_result = subprocess.run(
                        [compiler, "-std=c99", "-Wall", "-Wextra", "-Werror", "-pedantic",
                         str(code_path), str(test_file), "-o", str(binary)],
                        cwd=temp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    if compile_result.returncode:
                        outcomes[name] = (False, "compile failed:\n" + compile_result.stdout)
                        continue
                    proc = subprocess.run([str(binary)], cwd=temp, text=True,
                                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                outcomes[name] = (proc.returncode == 0, proc.stdout)
        preferred_pass = outcomes.get("preferred", (False, ""))[0]
        unpreferred_pass = outcomes.get("unpreferred", (False, ""))[0]
        status = preferred_pass and not unpreferred_pass
        print(f"{pair['pair_id']}: {'PASS' if status else 'FAIL'} "
              f"preferred={'pass' if preferred_pass else 'fail'} "
              f"unpreferred={'pass' if unpreferred_pass else 'fail'}")
        if not status:
            for name in ("preferred", "unpreferred"):
                if name in outcomes and outcomes[name][1]:
                    errors.append(f"{pair['pair_id']} {name} output:\n{outcomes[name][1]}")

    if len(pairs) != 50:
        errors.append(f"expected 50 pairs, found {len(pairs)}")
    if split_counts != Counter({"build": 35, "dev": 15}):
        errors.append(f"split counts must be build=35, dev=15; got {dict(split_counts)}")
    if language_counts != Counter({"python": 25, "c99": 25}):
        errors.append(f"language counts must be python=25, c99=25; got {dict(language_counts)}")
    if set(category_counts) != {"spec", "review", "api"}:
        errors.append(f"expected spec, review, and api categories; got {dict(category_counts)}")
    print(f"pairs={len(pairs)} categories={dict(sorted(category_counts.items()))} "
          f"splits={dict(sorted(split_counts.items()))} languages={dict(sorted(language_counts.items()))}")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("contrast checks passed: every preferred answer passes and every unpreferred answer fails")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
