"""Send a coding brief to the local model and write the returned code block to a file.

  python tools/delegate.py <brief.md> <output file> [--feedback <file>] [--name <label>] [--test CMD] [--attempts N]

Each attempt is logged under results/delegations/<UTC timestamp>-<label>/ (brief, raw reply, stats).
"""
import argparse
import json
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8080/v1/chat/completions"
SYSTEM = ("You are a senior software engineer (C and Python). Follow the brief exactly. Reply with ONE fenced code block "
          "containing the complete file, and nothing else.")
RULES_FILE = ROOT / "tools" / "helper_rules.md"
# The closing fence must start its own line, so code that itself contains ``` (e.g. this file) is not cut short.
CODE_BLOCK = re.compile(r"^```[a-zA-Z0-9]*\n(.*?)\n```[ \t]*$", re.DOTALL | re.MULTILINE)

MAX_TEST_OUTPUT_CHARS = 4000
TEST_TIMEOUT_SECONDS = 600
DEFAULT_ATTEMPTS = 3


def system_prompt(use_rules: bool = True) -> str:
    if use_rules and RULES_FILE.exists():
        return SYSTEM + "\n\n" + RULES_FILE.read_text()
    return SYSTEM


def ask(prompt: str, system: str = SYSTEM) -> tuple[str, dict, float]:
    body = {"model": "default_model", "max_tokens": 8192, "temperature": 0.2,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as resp:
        payload = json.load(resp)
    return payload["choices"][0]["message"]["content"] or "", payload.get("usage", {}), time.perf_counter() - start


def run_test(cmd: str) -> tuple[bool, str]:
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=TEST_TIMEOUT_SECONDS)  # runs in the caller's directory
    except subprocess.TimeoutExpired:
        return False, f"test command timed out after {TEST_TIMEOUT_SECONDS} s"
    output = proc.stdout + proc.stderr
    tail = output[-MAX_TEST_OUTPUT_CHARS:]
    return proc.returncode == 0, f"exit code {proc.returncode}\n{tail}"


def repair_prompt(brief: str, code: str, report: str) -> str:
    fence = chr(96) * 3
    return (brief + "\n\n## Your previous attempt failed\n\n"
            + "Previous code:\n" + fence + "\n" + code + fence + "\n\n"
            + "Test output:\n" + fence + "\n" + report.rstrip("\n") + "\n" + fence + "\n\n"
            + "Fix it. Reply with the complete corrected file.")


def current_adapter() -> str:
    path = ROOT / "results" / "delegations" / "current-adapter.txt"
    if path.exists():
        return path.read_text().strip()
    return ""


def delegate(brief: str, output: Path, test_cmd: str | None = None, attempts: int = DEFAULT_ATTEMPTS,
             name: str = "task", use_rules: bool = True) -> dict:
    utc = datetime.now(timezone.utc)
    log_dir = ROOT / "results" / "delegations" / f"{utc:%Y%m%dT%H%M%SZ}-{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "system.md").write_text(system_prompt(use_rules))

    prompt = brief
    total_secs = 0.0
    total_tokens = 0
    passed_first = None
    passed_final = None
    attempts_used = 0
    max_attempts = attempts if test_cmd else 1

    for n in range(1, max_attempts + 1):
        attempts_used = n
        reply, usage, secs = ask(prompt, system_prompt(use_rules))
        total_secs += secs
        total_tokens += usage.get("total_tokens", 0)

        attempt_dir = log_dir / f"attempt-{n}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        (attempt_dir / "prompt.md").write_text(prompt)
        (attempt_dir / "reply.md").write_text(reply)

        match = CODE_BLOCK.search(reply)
        if not match:
            passed = False
            report = "no code block in reply"
            code = ""
        else:
            code = match.group(1) + "\n"
            output.write_text(code)
            if test_cmd:
                passed, report = run_test(test_cmd)
            else:
                passed = None
                report = ""

        if n == 1:
            passed_first = passed
        passed_final = passed

        if test_cmd:
            (attempt_dir / "test.txt").write_text(report)

        if passed is None or passed is True:
            break
        prompt = repair_prompt(brief, code, report)

    result = {
        "name": name,
        "utc": utc.isoformat(),
        "attempts_used": attempts_used,
        "passed_first": passed_first,
        "passed_final": passed_final,
        "seconds": round(total_secs, 1),
        "tokens": total_tokens,
        "test_cmd": test_cmd,
        "log_dir": str(log_dir),
        "adapter": current_adapter(),
        "output": str(output),
    }
    (log_dir / "stats.json").write_text(json.dumps(result, indent=2))
    outcomes_path = ROOT / "results" / "delegations" / "outcomes.jsonl"
    with outcomes_path.open("a") as f:
        f.write(json.dumps(result) + "\n")
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("brief")
    p.add_argument("output")
    p.add_argument("--feedback", help="file with previous code + test output to fix")
    p.add_argument("--name", default="task")
    p.add_argument("--no-rules", action="store_true")
    p.add_argument("--test", dest="test_cmd", default=None, help="shell command to run after each attempt")
    p.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    a = p.parse_args()

    prompt = Path(a.brief).read_text()
    if a.feedback:
        prompt += "\n\n## Your previous attempt failed\n\n" + Path(a.feedback).read_text() + \
                  "\n\nFix it. Reply with the complete corrected file."

    result = delegate(prompt, Path(a.output), test_cmd=a.test_cmd, attempts=a.attempts,
                      name=a.name, use_rules=not a.no_rules)

    passed_final = result["passed_final"]
    status = "PASS" if passed_final else "FAIL" if passed_final is False else "untested"
    print(f"{a.output}: {status} after {result['attempts_used']} attempt(s), "
          f"{result['seconds']}s - log {result['log_dir']}")

    if passed_final is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
