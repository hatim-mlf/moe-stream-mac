"""Run a local-model write, test, and fix loop.

Usage: python tools/autofix.py <brief.md> <output> --test "<shell command>"
       [--max-attempts 4] [--name label] [--no-rules]
       [--functions name1 name2 ...] [--keep-on-fail] [--test-timeout 600]
"""
import argparse
import ast
import io
import json
import re
import subprocess
import sys
import time
import tokenize
from datetime import datetime, timezone
from pathlib import Path

try:
    from . import delegate
except ImportError:
    import delegate

ROOT = Path(__file__).resolve().parents[1]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask the local model to write, test, and fix code.")
    parser.add_argument("brief")
    parser.add_argument("output")
    parser.add_argument("--test", required=True, help="shell command that exits 0 when the code passes")
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--name", default="task")
    parser.add_argument("--no-rules", action="store_true")
    parser.add_argument("--functions", nargs="+", metavar="NAME")
    parser.add_argument("--keep-on-fail", action="store_true")
    parser.add_argument("--test-timeout", type=float, default=600)
    args = parser.parse_args(argv)
    if args.max_attempts < 1 or args.test_timeout <= 0:
        parser.error("--max-attempts and --test-timeout must be positive")
    return args


def safe_label(label: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-.")
    return cleaned or "task"


def newline_for(raw: bytes) -> bytes:
    return b"\r\n" if b"\r\n" in raw else b"\n"


def top_level_definitions(source: str) -> dict[str, list[ast.AST]]:
    tree = ast.parse(source)
    found: dict[str, list[ast.AST]] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            found.setdefault(node.name, []).append(node)
    return found


def node_start(node: ast.AST) -> int:
    decorators = getattr(node, "decorator_list", [])
    return min([node.lineno, *(item.lineno for item in decorators)])


def extract_functions(code: str, names: list[str], encoding: str = "utf-8") -> dict[str, bytes]:
    found = top_level_definitions(code)
    missing = [name for name in names if name not in found]
    duplicates = [name for name in names if len(found.get(name, [])) > 1]
    if missing or duplicates:
        details = []
        if missing:
            details.append("reply is missing top-level definition(s): " + ", ".join(missing))
        if duplicates:
            details.append("reply has duplicate top-level definition(s): " + ", ".join(duplicates))
        raise ValueError("; ".join(details))

    lines = code.splitlines(keepends=True)
    pieces = {}
    for name in names:
        node = found[name][0]
        segment = "".join(lines[node_start(node) - 1:node.end_lineno])
        pieces[name] = segment.encode(encoding)
    return pieces


def replace_functions(path: Path, code: str, names: list[str]) -> None:
    original = path.read_bytes()
    encoding, _ = tokenize.detect_encoding(io.BytesIO(original).readline)
    source = original.decode(encoding)
    existing = top_level_definitions(source)
    missing = [name for name in names if name not in existing]
    if missing:
        raise ValueError("output is missing top-level definition(s): " + ", ".join(missing))
    replacements = extract_functions(code, names, encoding)
    old_lines = original.splitlines(keepends=True)
    eol = newline_for(original)
    edits = []
    for name in names:
        for node in existing[name]:
            start, end = node_start(node), node.end_lineno
            replacement = replacements[name].replace(b"\r\n", b"\n").replace(b"\n", eol)
            if not replacement.endswith((b"\n", b"\r")):
                replacement += eol
            edits.append((start - 1, end, replacement))
    for start, end, replacement in sorted(edits, reverse=True):
        old_lines[start:end] = [replacement]
    path.write_bytes(b"".join(old_lines))


def write_code(path: Path, code: str, names: list[str] | None) -> None:
    if names:
        replace_functions(path, code, names)
    else:
        path.write_text(code.rstrip("\r\n") + "\n", encoding="utf-8")


def run_test(command: str, timeout: float) -> tuple[int, str]:
    try:
        result = subprocess.run(command, shell=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout)
        return result.returncode, result.stdout or ""
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, f"{output}\nTest timed out after {timeout:g} seconds."
    except OSError as exc:
        return 127, f"Could not run test command: {exc}"


def neutralize(text: str) -> str:
    return text.replace("<|", "<\\|").replace("<think>", "<think_").replace("</think>", "</think_")


def failure_prompt(brief: str, attempt: int, code: str, output: str) -> str:
    tail = "\n".join(output.splitlines()[-80:])
    return (f"{brief}\n\n## Attempt {attempt} failed\n\nYour code:\n```\n"
            f"{neutralize(code)}\n```\n\nTest output (last 80 lines):\n```\n"
            f"{neutralize(tail)}\n```\n\nFix it. Apply every point; change nothing else.")


def log_attempt(log_root: Path, label: str, number: int, prompt: str,
                reply: str, output: str, seconds: float, usage: dict,
                exit_code: int | None) -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    folder = log_root / f"{stamp}-{label}-a{number}"
    folder.mkdir(parents=True)
    (folder / "prompt.md").write_text(prompt, encoding="utf-8")
    (folder / "reply.md").write_text(reply, encoding="utf-8")
    (folder / "test_output.txt").write_text(output, encoding="utf-8")
    stats = {"seconds": round(seconds, 3), "usage": usage, "exit_code": exit_code}
    (folder / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def perform_attempt(prompt: str, output_path: Path, names: list[str] | None,
                    test: str, timeout: float, log_root: Path,
                    label: str, number: int, system: str) -> dict:
    started = time.perf_counter()
    try:
        reply, usage, seconds = delegate.ask(prompt, system)
        match = delegate.CODE_BLOCK.search(reply)
        code = match.group(1) if match else ""
        if not match:
            exit_code, output = None, "Failure: model reply did not contain a fenced code block."
        else:
            try:
                write_code(output_path, code, names)
            except (SyntaxError, UnicodeError, ValueError) as exc:
                exit_code, output = None, f"Could not apply returned code: {exc}"
            else:
                exit_code, output = run_test(test, timeout)
    except Exception as exc:
        reply, code, usage = "", "", {}
        seconds = time.perf_counter() - started
        exit_code, output = None, f"Model request failed: {type(exc).__name__}: {exc}"
    log_attempt(log_root, label, number, prompt, reply, output, seconds, usage or {}, exit_code)
    return {"passed": exit_code == 0, "code": code, "output": output, "exit_code": exit_code}


def restore_output(path: Path, original: bytes | None) -> None:
    if original is None:
        path.unlink(missing_ok=True)
    else:
        path.write_bytes(original)


def append_run(log_root: Path, label: str, attempts: int, passed: bool, total: float) -> None:
    log_root.mkdir(parents=True, exist_ok=True)
    record = {"label": label, "attempts": attempts, "passed": passed, "total_seconds": round(total, 3)}
    with (log_root / "autofix.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record) + "\n")


def system_for(use_rules: bool, names: list[str] | None) -> str:
    system = delegate.system_prompt(use_rules)
    if names:
        named = ", ".join(names)
        system = system.replace("containing the complete file, and nothing else.",
                                "containing only the named definitions, and nothing else.", 1)
        system += (f"\n\nFor this request, the code block must contain only the top-level "
                   f"function or class definitions named {named}. Do not return the complete file "
                   "or any other definitions.")
    return system


def run_autofix(args: argparse.Namespace) -> int:
    brief = Path(args.brief).read_text(encoding="utf-8")
    output_path = Path(args.output)
    original = output_path.read_bytes() if output_path.exists() else None
    log_root = ROOT / "results" / "delegations"
    label = safe_label(args.name)
    system = system_for(not args.no_rules, args.functions)
    prompt = brief
    start = time.perf_counter()
    passed, attempts = False, 0
    for number in range(1, args.max_attempts + 1):
        attempts = number
        result = perform_attempt(prompt, output_path, args.functions, args.test,
                                 args.test_timeout, log_root, label, number, system)
        if result["passed"]:
            passed = True
            break
        if number < args.max_attempts:
            prompt = failure_prompt(brief, number, result["code"], result["output"])
    if not passed and not args.keep_on_fail:
        restore_output(output_path, original)
    append_run(log_root, label, attempts, passed, time.perf_counter() - start)
    state = "passed" if passed else "failed"
    print(f"autofix {state} after {attempts} attempt(s): {args.output}")
    return 0 if passed else 1


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    return run_autofix(args)


if __name__ == "__main__":
    raise SystemExit(main())
