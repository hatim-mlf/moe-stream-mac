#!/usr/bin/env python3
"""Frozen task and public-mini evaluation for the local MLX OpenAI server."""
import argparse
import hashlib
import json
import random
import re
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8080/v1/chat/completions"
MANIFEST = json.loads((ROOT / "tools/ft_eval_manifest.json").read_text())
FROZEN_REVISION = MANIFEST["benchmark_revision"]
SYSTEM = ("You are a senior software engineer (C and Python). Follow the brief exactly. "
          "Reply with ONE fenced code block containing the requested code, and nothing else.\n\n" +
          (ROOT / "tools/helper_rules.md").read_text())
if hashlib.sha256(SYSTEM.encode()).hexdigest() != MANIFEST["system_sha256"]:
    raise RuntimeError("Frozen system prompt changed; create a new evaluation version intentionally")
FROZEN = [(t["id"], t["target"], t["first_attempt"], t["review_attempt"]) for t in MANIFEST["tasks"]]
PUBLIC_TASKS = [
    {"id": "HumanEval/5", "entry": "intersperse", "prompt": "from typing import List\n\ndef intersperse(numbers: List[int], delimeter: int) -> List[int]:\n    \"\"\"Insert delimeter between every pair of consecutive values.\"\"\"\n", "tests": "from candidate import intersperse\nassert intersperse([], 7) == []\nassert intersperse([5], 7) == [5]\nassert intersperse([1, 2, 3], 9) == [1, 9, 2, 9, 3]\nassert intersperse([2, 2, 2], 2) == [2, 2, 2, 2, 2]\n"},
    {"id": "HumanEval/7", "entry": "filter_by_substring", "prompt": "from typing import List\n\ndef filter_by_substring(strings: List[str], substring: str) -> List[str]:\n    \"\"\"Keep matching strings in their original order.\"\"\"\n", "tests": "from candidate import filter_by_substring\nassert filter_by_substring([], 'x') == []\nassert filter_by_substring(['abc', 'cab', 'xyz', 'abc'], 'ab') == ['abc', 'cab', 'abc']\nassert filter_by_substring(['A', 'a'], 'a') == ['a']\nassert filter_by_substring(['x', 'y'], '') == ['x', 'y']\n"},
    {"id": "HumanEval/58", "entry": "common", "prompt": "from typing import List\n\ndef common(l1: List[int], l2: List[int]) -> List[int]:\n    \"\"\"Return sorted unique integers present in both lists.\"\"\"\n", "tests": "from candidate import common\nassert common([1, 4, 3, 34, 653, 2, 5], [5, 7, 1, 5, 9, 653, 121]) == [1, 5, 653]\nassert common([5, 3, 2, 8], [3, 2]) == [2, 3]\nassert common([], [1]) == []\nassert common([4, 4, 2], [2, 4, 4]) == [2, 4]\n"},
]
PUBLIC_REFERENCES = {
    "HumanEval/5": "def intersperse(numbers, delimeter):\n    out = []\n    for index, value in enumerate(numbers):\n        if index:\n            out.append(delimeter)\n        out.append(value)\n    return out\n",
    "HumanEval/7": "def filter_by_substring(strings, substring):\n    return [value for value in strings if substring in value]\n",
    "HumanEval/58": "def common(l1, l2):\n    return sorted(set(l1) & set(l2))\n",
}
PREDICTIVE_TEST = """import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1])
from predictive import PredictivePrefetcher
class Reader:
    def read_sorted(self, layer, proj, ids):
        ids = np.asarray(ids, dtype=np.int64)
        return {name: (ids[:, None] + i).astype(np.float32) for i, name in enumerate(("weight", "scales", "biases"))}
pf = PredictivePrefetcher(Reader(), ("gate_proj", "up_proj"))
pf.request(4, np.array([1, 3, 5]))
got = pf.take(4, np.array([3, 4]))
assert sorted(got) == [3, 4]
assert all(isinstance(k, int) for k in got)
assert [int(x[0]) for x in got[3]["gate_proj"]] == [3, 4, 5]
assert [int(x[0]) for x in got[4]["up_proj"]] == [4, 5, 6]
assert pf.stats() == {"predicted_hits": 1, "misses": 1, "hit_rate": 0.5}
assert 4 not in pf._pending
assert sorted(pf.take(9, np.array([8])) ) == [8]
print("PREDICTIVE MINI TESTS PASSED")
"""


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", choices=["heldout", "public", "heldout-v2"], default="heldout")
    p.add_argument("--endpoint", default=API)
    p.add_argument("--model", default="default_model", help="OpenAI model selector passed to the MLX server")
    p.add_argument("--adapter", help="MLX server adapter path; sent as OpenAI `adapters` field")
    p.add_argument("--attempts", type=int, default=1, help="attempts with test feedback, 1 disables retries")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int)
    p.add_argument("--samples", type=int, default=1, help="independent model runs per task")
    p.add_argument("--review-notes", action="store_true", help="run the frozen held-out review prompts once")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", type=Path)
    p.add_argument("--self-check", action="store_true", help="test accepted repository code without a model")
    p.add_argument("--mutant-check", action="store_true", help="require every heldout-v2 mutant to fail")
    p.add_argument("--paired", nargs=2, type=Path, metavar=("BASE.json", "ADAPTER.json"))
    return p.parse_args(argv)


def extract_code(reply):
    match = re.search(r"^```[A-Za-z0-9_+-]*\s*\n(.*?)\n```\s*$", reply, re.S | re.M)
    return match.group(1) if match else ""


def request_model(prompt, endpoint, model, adapter, temperature=0.0, seed=None):
    body = {"model": model, "temperature": temperature, "max_tokens": 8192,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
    if adapter:
        body["adapters"] = adapter
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(endpoint, json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as response:
            return json.load(response)["choices"][0]["message"].get("content") or ""
    except OSError as error:
        raise RuntimeError(f"Cannot reach MLX server at {endpoint}: {error}") from error


def checkout_tree(destination):
    archive = subprocess.run(["git", "archive", "--format=tar", FROZEN_REVISION], cwd=ROOT,
                             check=True, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=__import__("io").BytesIO(archive.stdout)) as source:
        source.extractall(destination, filter="data")


def run_command(command, cwd, timeout=180):
    try:
        result = subprocess.run(command, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        output = error.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, f"Timed out after {timeout}s\n{output[-12000:]}"
    return result.returncode, result.stdout[-12000:]


def verify_v2_manifest(root=ROOT, manifest=None):
    """Refuse heldout-v2 execution if any frozen prompt, task, or test asset changed."""
    import sys
    tools_dir = str(root / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    from ft_heldout_v2_freeze import collect_manifest

    expected = manifest if manifest is not None else json.loads((root / "tools/ft_eval_heldout_v2.json").read_text())
    actual = collect_manifest(root)
    if actual != expected:
        raise RuntimeError("heldout-v2 frozen manifest mismatch; run tools/ft_heldout_v2_freeze.py only after intentional suite changes")
    return expected


def v2_task_records(review=False, root=ROOT):
    manifest = verify_v2_manifest(root)
    records = []
    for entry in manifest["tasks"]:
        folder = root / "evals" / "heldout-v2" / entry["id"]
        task = json.loads((folder / "task.json").read_text())
        task["_folder"] = str(folder)
        task["_manifest_entry"] = entry
        task["_prompt"] = (folder / ("review.md" if review else "spec.md")).read_text()
        records.append(task)
    return records


def run_v2_candidate_check(task, code):
    folder = Path(task["_folder"])
    with tempfile.TemporaryDirectory(prefix="ft-v2-") as work:
        tree = Path(work)
        shutil.copytree(folder, tree, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("reference", "mutants"))
        target = tree / task["target"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code.rstrip() + "\n")

        def expand(command):
            return [item.replace("{python}", str(ROOT / ".venv/bin/python")).replace("{cc}", "cc")
                    for item in command]

        if task["build_cmd"]:
            status, output = run_command(expand(task["build_cmd"]), tree, task["timeout_s"])
            if status:
                return status, "build failed:\n" + output
        return run_command(expand(task["test_cmd"]), tree, task["timeout_s"])


def self_check_v2_tasks(tasks):
    passed_count = 0
    for task in tasks:
        code = (Path(task["_folder"]) / "reference" / task["target"]).read_text()
        result = run_v2_candidate_check(task, code)
        passed = result[0] == 0
        print(f"{task['id']}: {'PASS' if passed else 'FAIL'}")
        if not passed:
            print(result[1])
        passed_count += int(passed)
    print(f"offline checks: {passed_count}/{len(tasks)} reference suites passed")
    return 0 if passed_count == len(tasks) else 1


def mutant_check_v2_tasks(tasks):
    checked = rejected_count = 0
    compile_errors = survivors = 0
    print("task_id\tmutant\tresult")
    for task in tasks:
        folder = Path(task["_folder"])
        mutant_dirs = sorted(path for path in (folder / "mutants").iterdir() if path.is_dir())
        for mutant_dir in mutant_dirs:
            checked += 1
            code = (mutant_dir / task["target"]).read_text()
            result = run_v2_candidate_check(task, code)
            build_failed = result[1].startswith("build failed:")
            rejected = result[0] != 0 and not build_failed
            rejected_count += int(rejected)
            compile_errors += int(build_failed)
            survivors += int(result[0] == 0)
            label = "COMPILE_ERROR" if build_failed else "REJECTED" if rejected else "SURVIVED"
            print(f"{task['id']}\t{mutant_dir.name}\t{label}")
            if build_failed or not rejected:
                print(result[1])
    print(f"mutant checks: {rejected_count}/{checked} rejected by tests; compile_errors={compile_errors}, survivors={survivors}")
    return 0 if rejected_count == checked else 1


def make_predictive_check(tree):
    path = tree / "ft_predictive_check.py"
    path.write_text(PREDICTIVE_TEST)
    return [str(ROOT / ".venv/bin/python"), str(path), str(tree / "engine/hybrid")]


def run_task_check(task_id, tree):
    engine = tree / "engine"
    if task_id in {"qmat", "qmat-row"}:
        built = run_command(["make", "-C", str(engine), "build/test_qmat"], tree)
        if built[0]:
            return built
        files = sorted((engine / "tests/fixtures").glob("qmat_*.bin"))
        return run_command([str(engine / "build/test_qmat"), *map(str, files)], tree)
    if task_id in {"ops", "gdn", "qgroup"}:
        binary = {"ops": "test_ops", "gdn": "test_gdn", "qgroup": "test_qgroup"}[task_id]
        built = run_command(["make", "-C", str(engine), f"build/{binary}"], tree)
        if built[0]:
            return built
        tail = [str(engine / "tests/fixtures")] if task_id in {"ops", "gdn"} else []
        return run_command([str(engine / "build" / binary), *tail], tree)
    if task_id in {"expert-cache", "pinned"}:
        test = "test_expert_cache.py" if task_id == "expert-cache" else "test_pinned.py"
        return run_command([str(ROOT / ".venv/bin/python"), str(engine / "hybrid" / test)], tree)
    if task_id == "predictive":
        return run_command(make_predictive_check(tree), tree)
    raise ValueError(f"No offline test harness for {task_id}")


def install_and_check(task_id, filename, code, public_test=None):
    with tempfile.TemporaryDirectory(prefix="ft-eval-") as work:
        tree = Path(work)
        if public_test:
            candidate = tree / "candidate.py"
            candidate.write_text(code + "\n")
            checker = tree / "test_public.py"
            checker.write_text(public_test)
            return run_command([str(ROOT / ".venv/bin/python"), str(checker)], tree)
        checkout_tree(tree)
        target = tree / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(code + "\n")
        return run_task_check(task_id, tree)


def task_prompt(task, review):
    folder = task[3] if review else task[2]
    field = "review_sha256" if review else "first_sha256"
    expected = next(entry[field] for entry in MANIFEST["tasks"] if entry["id"] == task[0])
    path = ROOT / "results/delegations" / folder / "prompt.md"
    content = path.read_text()
    if hashlib.sha256(content.encode()).hexdigest() != expected:
        raise RuntimeError(f"Frozen prompt changed: {path}")
    return content


def task_records(split, review):
    if split == "public":
        if review:
            raise ValueError("Public HumanEval tasks have no project review notes")
        return PUBLIC_TASKS
    if split == "heldout-v2":
        return v2_task_records(review)
    return FROZEN


def evaluate_one(task, args, sample_index=0):
    public = args.split == "public"
    v2 = args.split == "heldout-v2"
    task_id = task["id"] if public or v2 else task[0]
    filename = "candidate.py" if public else (task["target"] if v2 else task[1])
    prompt = task["prompt"] if public else (task["_prompt"] if v2 else task_prompt(task, args.review_notes))
    last_code, last_output, attempts, status = "", "", 0, 1
    for number in range(1, args.attempts + 1):
        attempts = number
        sample_seed = None if args.seed is None else args.seed + sample_index
        reply = request_model(prompt, args.endpoint, args.model, args.adapter, args.temperature, sample_seed)
        last_code = extract_code(reply)
        if not last_code:
            last_output = "No single fenced code block found in model response."
            break
        if v2:
            status, last_output = run_v2_candidate_check(task, last_code)
        elif public:
            test = task["tests"].replace("from candidate import", "from candidate import")
            status, last_output = install_and_check(task_id, filename, last_code, test)
        else:
            status, last_output = install_and_check(task_id, filename, last_code)
        if status == 0:
            break
        prompt += (f"\n\n## Attempt {number} failed\n\nYour code:\n```\n{last_code}\n```"
                   f"\n\nTest output:\n```\n{last_output}\n```\n\nFix every reported issue; change nothing else.")
    passed = bool(last_code) and status == 0 if "status" in locals() else False
    return {"task_id": task_id, "sample_index": sample_index, "passed": passed,
            "first_pass": bool(passed and attempts == 1), "attempts": attempts,
            "attempt_score": attempts if passed else args.attempts + 1, "output_tail": last_output}


def load_accepted(path):
    result = subprocess.run(["git", "show", f"{FROZEN_REVISION}:{path}"], cwd=ROOT, text=True,
                            stdout=subprocess.PIPE, check=True)
    return result.stdout


def self_check(task, public=False):
    if public:
        return install_and_check(task["id"], "candidate.py", PUBLIC_REFERENCES[task["id"]], task["tests"])
    task_id, filename = task[0], task[1]
    code = load_accepted(filename)
    return install_and_check(task_id, filename, code)


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * fraction))]


def bootstrap_ci(differences, seed=20261104, samples=10000):
    rng = random.Random(seed)
    means = [sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(samples)]
    return percentile(means, 0.025), percentile(means, 0.975)


def paired_report(paths):
    base, adapter = [json.loads(path.read_text()) for path in paths]
    left = {r["task_id"]: r for r in base["results"]}
    right = {r["task_id"]: r for r in adapter["results"]}
    if len(left) != len(base["results"]) or len(right) != len(adapter["results"]):
        raise ValueError("paired result files must not contain duplicate task IDs")
    common = sorted(left.keys() & right.keys())
    if (base["split"], base["attempt_cap"], base["review_notes"]) != (adapter["split"], adapter["attempt_cap"], adapter["review_notes"]):
        raise ValueError("paired files must use the same split, retry cap, and review-note mode")
    left_suite = base.get("suite", base["split"])
    right_suite = adapter.get("suite", adapter["split"])
    left_samples = base.get("samples", 1)
    right_samples = adapter.get("samples", 1)
    if (left_suite != right_suite or base.get("suite_sha256") != adapter.get("suite_sha256") or
            base["benchmark_revision"] != adapter["benchmark_revision"] or
            base.get("temperature", 0) != adapter.get("temperature", 0) or left_samples != right_samples):
        raise ValueError("paired files must use the same suite, frozen hashes, temperature, and sample count")
    if base["model"] != adapter["model"]:
        raise ValueError("paired files must use the same frozen suite and base model")
    if not common:
        raise ValueError("paired result files have no task ids in common")
    if left.keys() != right.keys():
        raise ValueError("paired files must contain exactly the same task IDs")
    for field in ("first_pass", "passed"):
        left_values = [float(left[k][field]) for k in common]
        right_values = [float(right[k][field]) for k in common]
        differences = [r - l for l, r in zip(left_values, right_values)]
        print(f"{field}: base={sum(left_values)/len(common):.3f}, adapter={sum(right_values)/len(common):.3f}, "
              f"delta={sum(differences)/len(common):+.3f}, paired-bootstrap-95%={bootstrap_ci(differences)}, n_tasks={len(common)}")
    deltas = [right[k]["attempt_score"] - left[k]["attempt_score"] for k in common]
    left_attempts = [float(left[k]["attempt_score"]) for k in common]
    right_attempts = [float(right[k]["attempt_score"]) for k in common]
    print(f"attempt_score (lower better): base={sum(left_attempts)/len(common):.3f}, "
          f"adapter={sum(right_attempts)/len(common):.3f}, delta={sum(deltas)/len(common):+.3f}, "
          f"paired-bootstrap-95%={bootstrap_ci(deltas)}, n_tasks={len(common)}")


def save_results(results, args):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = args.out or ROOT / "results/finetune" / f"{args.split}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    suite_hash = verify_v2_manifest()["suite_sha256"] if args.split == "heldout-v2" else None
    payload = {"benchmark_revision": suite_hash or FROZEN_REVISION, "split": args.split,
               "suite": "heldout-v2" if args.split == "heldout-v2" else args.split,
               "model": args.model, "adapter": args.adapter or "base", "attempt_cap": args.attempts,
               "endpoint": args.endpoint, "temperature": args.temperature, "seed": args.seed,
               "samples": args.samples, "max_tokens": 8192,
               "review_notes": args.review_notes, "results": results}
    if suite_hash:
        payload["suite_sha256"] = suite_hash
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def self_check_tasks(tasks, split):
    outcomes = []
    for task in tasks:
        result = self_check(task, split == "public")
        task_id = task["id"] if split == "public" else task[0]
        passed = result is not None and result[0] == 0
        label = "PASS" if passed else "FAIL" if result else "no offline fixture"
        print(f"{task_id}: {label}")
        if result and result[0] != 0:
            print(result[1])
        if passed:
            outcomes.append(task_id)
    print(f"offline checks: {len(outcomes)}/{len(tasks)} accepted-code suites passed")
    return 0 if len(outcomes) == len(tasks) else 1


def aggregate_samples(task_id, samples):
    count = len(samples)
    return {"task_id": task_id,
            "samples": [dict(row, sample_index=i) for i, row in enumerate(samples)],
            "first_pass": sum(float(row["first_pass"]) for row in samples) / count,
            "passed": sum(float(row["passed"]) for row in samples) / count,
            "attempts": sum(row["attempts"] for row in samples) / count,
            "attempt_score": sum(row["attempt_score"] for row in samples) / count,
            "output_tail": samples[-1]["output_tail"]}


def run_live_evaluation(tasks, args):
    try:
        results = []
        for task in tasks:
            task_id = task["id"] if args.split in {"public", "heldout-v2"} else task[0]
            sample_rows = [evaluate_one(task, args, index) for index in range(args.samples)]
            row = sample_rows[0] if args.samples == 1 else aggregate_samples(task_id, sample_rows)
            if args.samples == 1:
                row.pop("sample_index", None)
            results.append(row)
    except RuntimeError as error:
        raise SystemExit(str(error)) from error
    for row in results:
        print(f"{row['task_id']}: first={row['first_pass']} pass={row['passed']} attempts={row['attempts']}")
    path = save_results(results, args)
    first = sum(row["first_pass"] for row in results) / max(1, len(results))
    passed = sum(row["passed"] for row in results) / max(1, len(results))
    all_samples = [sample for row in results for sample in row.get("samples", [row])]
    solved = [row["attempts"] for row in all_samples if row["passed"]]
    mean_attempts = sum(solved) / len(solved) if solved else None
    censored = sum(row["attempt_score"] for row in all_samples) / max(1, len(all_samples))
    print(f"first_attempt_pass={first:.3f} pass_at_cap={passed:.3f} "
          f"mean_attempts_to_pass={mean_attempts} censored_attempt_score={censored:.3f} output={path}")
    return 0


def main(argv=None):
    args = arguments(argv)
    if args.paired:
        paired_report(args.paired)
        return 0
    if args.attempts < 1:
        raise SystemExit("--attempts must be >= 1")
    if args.samples < 1:
        raise SystemExit("--samples must be >= 1")
    if args.temperature < 0:
        raise SystemExit("--temperature must be >= 0")
    if args.seed is not None and args.seed < 0:
        raise SystemExit("--seed must be >= 0")
    if args.mutant_check and args.split != "heldout-v2":
        raise SystemExit("--mutant-check applies only to --split heldout-v2")
    if args.split == "public" and args.review_notes:
        raise SystemExit("--review-notes applies only to the repository held-out split")
    tasks = task_records(args.split, args.review_notes)
    if args.limit:
        tasks = tasks[:args.limit]
    if args.mutant_check:
        return mutant_check_v2_tasks(tasks)
    if args.self_check:
        if args.split == "heldout-v2":
            return self_check_v2_tasks(tasks)
        return self_check_tasks(tasks, args.split)
    if args.review_notes and args.attempts != 1:
        raise SystemExit("review-note compliance runs one response per frozen note")
    return run_live_evaluation(tasks, args)


if __name__ == "__main__":
    raise SystemExit(main())
