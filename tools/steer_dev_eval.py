import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ft_eval


def load_pairs(split: str) -> list[dict]:
    path = ROOT / "data" / "steering" / "pairs.jsonl"
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            if obj.get("split") == split:
                rows.append(obj)
    if not rows:
        raise ValueError(f"No pairs found for split '{split}'")
    return rows


def run_candidate(task_id: str, code: str) -> tuple[bool, str]:
    if not code:
        return (False, "no code block")
    with tempfile.TemporaryDirectory() as temp:
        test_file = None
        lang = "python"
        test_py = ROOT / "data" / "steering" / "tests" / task_id / "test.py"
        test_c = ROOT / "data" / "steering" / "tests" / task_id / "test.c"
        if test_py.exists():
            test_file = test_py
        elif test_c.exists():
            test_file = test_c
            lang = "c"
        else:
            return (False, "no test file found")
        code_path = os.path.join(temp, "candidate.py" if lang == "python" else "candidate.c")
        with open(code_path, "w", encoding="utf-8") as fh:
            fh.write(code + "\n")
        if lang == "python":
            env = os.environ.copy()
            pypath = env.get("PYTHONPATH", "")
            if pypath:
                env["PYTHONPATH"] = temp + os.pathsep + pypath
            else:
                env["PYTHONPATH"] = temp
            result = subprocess.run(
                [sys.executable, str(test_file)],
                cwd=temp,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            return (result.returncode == 0, result.stdout)
        else:
            cc = shutil.which("cc")
            if not cc:
                return (False, "cc not found")
            check_path = os.path.join(temp, "check")
            compile_args = [
                cc,
                "-std=c99",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                code_path,
                str(test_file),
                "-o",
                check_path,
            ]
            comp_result = subprocess.run(
                compile_args,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            if comp_result.returncode != 0:
                return (False, "compile failed:\n" + comp_result.stdout)
            run_result = subprocess.run(
                [check_path],
                cwd=temp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=60,
            )
            return (run_result.returncode == 0, run_result.stdout)


def ask_server(endpoint: str, messages: list) -> str:
    payload = {
        "model": "default_model",
        "temperature": 0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": messages,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return ""


def evaluate(pairs: list[dict], ask) -> list[dict]:
    rows = []
    for pair in pairs:
        reply = ask(pair["messages"])
        code = ft_eval.extract_code(reply)
        task_id = pair["task_id"]
        passed, output = run_candidate(task_id, code)
        output_tail = output[-2000:] if output else ""
        rows.append(
            {
                "task_id": task_id,
                "category": pair["category"],
                "passed": passed,
                "output_tail": output_tail,
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="dev")
    parser.add_argument("--endpoint", default=ft_eval.API)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    pairs = load_pairs(args.split)
    rows = evaluate(pairs, lambda m: ask_server(args.endpoint, m))
    passed_count = sum(1 for r in rows if r["passed"])
    output_obj = {
        "label": args.label,
        "split": args.split,
        "passed": passed_count,
        "total": len(rows),
        "rows": rows,
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(output_obj, fh, indent=1)
    print(f"{args.label}: {passed_count}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())