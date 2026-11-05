#!/usr/bin/env python3
"""Run reference and mutant checks for a directory of supervised coding tasks."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {"c": ".c", "python": ".py"}
MAX_BRIEF_SOLUTION_CHARS = 6000
MAX_ROW_BYTES = 7500


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--only", metavar="ID")
    return parser.parse_args(argv)


def load_runner():
    tools_dir = str(ROOT / "tools")
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
    import ft_eval
    import ft_build_dataset
    return ft_eval, ft_build_dataset


def python_with_test_dependencies(ft_eval):
    preferred = ft_eval.ROOT / ".venv/bin/python"
    candidates = [preferred, Path(sys.executable), Path("/opt/anaconda3/bin/python")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        result = subprocess.run([str(candidate), "-c", "import numpy, pytest"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if result.returncode == 0:
            return candidate
    return None


def read_task_folders(directory):
    folders = sorted(path for path in directory.iterdir() if path.is_dir())
    ids = []
    for folder in folders:
        meta_path = folder / "meta.json"
        if not meta_path.is_file():
            raise ValueError(f"{folder}: missing meta.json")
        meta = json.loads(meta_path.read_text())
        task_id = meta.get("id")
        if not isinstance(task_id, str) or not task_id or task_id != folder.name:
            raise ValueError(f"{meta_path}: id must be a nonempty string matching its folder")
        ids.append(task_id)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return folders


def validate_meta(folder, meta):
    language = meta.get("language")
    kind = meta.get("kind")
    if language not in EXTENSIONS:
        raise ValueError(f"{folder}: language must be c or python")
    if kind not in {"implement", "skeleton", "review"}:
        raise ValueError(f"{folder}: invalid kind {kind!r}")
    if not isinstance(meta.get("category"), str) or not meta["category"].strip():
        raise ValueError(f"{folder}: category must be nonempty")
    if meta.get("source") != "codex-batch1":
        raise ValueError(f"{folder}: source must be 'codex-batch1'")
    if kind == "skeleton" and not (folder / f"skeleton{EXTENSIONS[language]}").is_file():
        raise ValueError(f"{folder}: skeleton task is missing its skeleton file")
    if kind == "review":
        notes = meta.get("review_notes")
        previous = meta.get("previous_attempt")
        if not isinstance(notes, list) or not notes or not all(isinstance(n, str) and n.strip() for n in notes):
            raise ValueError(f"{folder}: review task needs nonempty review_notes")
        if not isinstance(previous, str) or not (folder / previous).is_file():
            raise ValueError(f"{folder}: review task needs a valid previous_attempt path")


def task_prompt(folder, meta):
    prompt = (folder / "spec.md").read_text().strip()
    language = meta["language"]
    if meta["kind"] == "skeleton":
        skeleton = (folder / f"skeleton{EXTENSIONS[language]}").read_text().strip()
        fence = "c" if language == "c" else "python"
        prompt += f"\n\n## Skeleton\n\n```{fence}\n{skeleton}\n```"
    if meta["kind"] == "review":
        attempt = (folder / meta["previous_attempt"]).read_text().strip()
        fence = "c" if language == "c" else "python"
        notes = "\n".join(f"- {note}" for note in meta["review_notes"])
        prompt += (f"\n\n## Previous attempt\n\n```{fence}\n{attempt}\n```"
                   f"\n\n## Review notes\n\n{notes}")
    return prompt


def row_size_ok(prompt, solution, system):
    content = "\n".join((system, prompt, solution))
    return len(content.encode("utf-8")) <= MAX_ROW_BYTES


def runner_task(folder, language):
    target = "candidate.c" if language == "c" else "candidate.py"
    task = {"_folder": str(folder), "target": target, "timeout_s": 20,
            "build_cmd": [], "test_cmd": []}
    tests = sorted((folder / "tests").glob("test_task.*"))
    if len(tests) != 1:
        raise ValueError(f"{folder}: expected exactly one tests/test_task.c or .py")
    if language == "c":
        if tests[0].suffix != ".c":
            raise ValueError(f"{folder}: C task needs tests/test_task.c")
        task["build_cmd"] = ["{cc}", "-std=c99", "-Wall", "-Wextra", "-Werror", "-O1",
                             "tests/test_task.c", "-lm", "-o", "check.bin"]
        task["test_cmd"] = ["./check.bin"]
    else:
        if tests[0].suffix != ".py":
            raise ValueError(f"{folder}: Python task needs tests/test_task.py")
        task["test_cmd"] = ["{python}", "-m", "pytest", "-q", "tests/test_task.py"]
    return task


def run_candidate(ft_eval, task, source, test_python):
    if test_python is None or test_python == ft_eval.ROOT / ".venv/bin/python":
        return ft_eval.run_v2_candidate_check(task, source)
    with tempfile.TemporaryDirectory(prefix="ft-python-runner-") as folder:
        shim_root = Path(folder)
        python_link = shim_root / ".venv/bin/python"
        python_link.parent.mkdir(parents=True)
        python_link.symlink_to(test_python)
        original_root = ft_eval.ROOT
        ft_eval.ROOT = shim_root
        try:
            return ft_eval.run_v2_candidate_check(task, source)
        finally:
            ft_eval.ROOT = original_root


def check_folder(folder, ft_eval, ft_build_dataset, test_python):
    meta = json.loads((folder / "meta.json").read_text())
    validate_meta(folder, meta)
    spec_path = folder / "spec.md"
    solution_path = folder / f"solution{EXTENSIONS[meta['language']]}"
    mutants_dir = folder / "mutants"
    tests_dir = folder / "tests"
    for path in (spec_path, solution_path, mutants_dir, tests_dir):
        if not path.exists():
            raise ValueError(f"{folder}: missing required path {path.name}")
    solution = solution_path.read_text()
    spec = spec_path.read_text()
    chars = len(spec) + len(solution)
    if chars > MAX_BRIEF_SOLUTION_CHARS:
        raise ValueError(f"{folder}: spec + reference is {chars} characters (limit {MAX_BRIEF_SOLUTION_CHARS})")
    prompt = task_prompt(folder, meta)
    fence = "c" if meta["language"] == "c" else "python"
    response = ft_build_dataset.code_fence(solution, fence)
    if not row_size_ok(prompt, response, ft_eval.SYSTEM):
        raise ValueError(f"{folder}: estimated training row exceeds {MAX_ROW_BYTES} UTF-8 bytes")
    task = runner_task(folder, meta["language"])
    reference_status, reference_output = run_candidate(ft_eval, task, solution, test_python)
    mutant_paths = sorted(path for path in mutants_dir.iterdir() if path.is_file())
    if not mutant_paths:
        raise ValueError(f"{folder}: mutants/ must contain at least one mutant")
    if any(path.suffix != EXTENSIONS[meta["language"]] for path in mutant_paths):
        raise ValueError(f"{folder}: mutant extension does not match language")
    mutant_results = [(path, run_candidate(ft_eval, task, path.read_text(), test_python))
                      for path in mutant_paths]
    for path, _ in mutant_results:
        if path.stat().st_size > MAX_BRIEF_SOLUTION_CHARS:
            raise ValueError(f"{path}: mutant exceeds size limit")
    return {"id": meta["id"], "reference_status": reference_status,
            "reference_output": reference_output, "mutants": mutant_results,
            "chars": chars, "category": meta["category"], "kind": meta["kind"]}


def main(argv=None):
    args = arguments(argv)
    directory = args.directory.resolve()
    if not directory.is_dir():
        raise SystemExit(f"task directory does not exist: {directory}")
    try:
        folders = read_task_folders(directory)
        if args.only:
            folders = [folder for folder in folders if folder.name == args.only]
            if not folders:
                raise ValueError(f"task id not found: {args.only}")
        ft_eval, ft_build_dataset = load_runner()
        test_python = python_with_test_dependencies(ft_eval)
        results = [check_folder(folder, ft_eval, ft_build_dataset, test_python) for folder in folders]
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"task check error: {error}", file=sys.stderr)
        return 1
    print("task_id\tkind\tchars\treference\tmutants")
    failures = 0
    for result in results:
        ref_ok = result["reference_status"] == 0
        mutant_ok = all(status[0] != 0 for _, status in result["mutants"])
        failures += int(not ref_ok) + int(not mutant_ok)
        states = ",".join(f"{path.name}:{'FAIL' if status[0] == 0 else 'PASS'}"
                           for path, status in result["mutants"])
        print(f"{result['id']}\t{result['kind']}\t{result['chars']}\t"
              f"{'PASS' if ref_ok else 'FAIL'}\t{states}")
        if not ref_ok:
            print(result["reference_output"])
        for path, (status, output) in result["mutants"]:
            if status == 0:
                print(f"mutant unexpectedly passed: {path}")
                print(output)
    print(f"checked {len(results)} task(s); violations={failures}")
    return int(failures != 0)


if __name__ == "__main__":
    raise SystemExit(main())
