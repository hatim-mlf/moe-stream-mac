"""Eval B of brief 51: the 30 code-compare cases solved agentically through the pi coding agent.

  .venv/bin/python tools/pi_agent_eval.py run --arm pi --out results/<UTC>-pi-adapter/eval-b [--limit N]
  .venv/bin/python tools/pi_agent_eval.py run --arm base --out ...
  .venv/bin/python tools/pi_agent_eval.py summary --out ...

Each case gets a fresh temp workspace (statement, stub, public examples, run_tests.sh) and a private HOME
with pi's models.json, runs pi under sandbox-exec (no network except the local server, no reads of the
user's home, writes only inside the case root), then the final solution is scored on all hidden tests by
tools/qwen38_code_compare.py as model `agent-<arm>`. Resumable: cases with a record.json are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen38_code_compare as cc  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = ROOT / "results/20261108T000646Z-qwen38-code-comparison"
PORT = 8080
PI = Path.home() / ".local/pi/bin/pi"
NODE = Path.home() / ".local/bin/node"
ADAPTER = ROOT / "adapters/qwen38-pi-v1"
WALL_LIMIT_S = 1800
TOOL_CALL_LIMIT = 40
TEMPERATURE = 0.8  # the adapter card's evaluation sampling
TOP_P = 1.0
MAX_TOKENS = 8192
CONTEXT_WINDOW = 131072
EXAMPLE_TIME_FACTOR = 2  # visible examples get 2x the judge's time limit
KILL_GRACE_S = 10
# Caps the server's cross-request KV cache (eviction only; outputs unchanged). Without it the base arm's cache
# grew to 8.9 GB and Metal ran out of memory on the 32 GB Mac (brief 51, 2026-11-23 10:05 UTC).
PROMPT_CACHE_BYTES = 4 * 1024**3
# 2048-token prefill steps over a 45k-token context OOMed Metal (13:41 UTC); 512 cuts the attention scratch 4x.
PREFILL_STEP_SIZE = 512
SERVER_FATAL = ("ERROR", '" 404 -', '" 500 -')
KAT_MODEL = Path.home() / "models/KAT-Coder-V2.5-Dev-OptiQ-4bit"  # brief 52
ARMS = ("pi", "base", "kat")
ARM_MODELS = {"pi": cc.BASE_MODEL, "base": cc.BASE_MODEL, "kat": KAT_MODEL}
PAIRS = (("pi", "base"), ("kat", "base"), ("kat", "pi"))
STUBS = {"python3": "# TODO: read stdin, write stdout\n", "cpp17": "int main() { return 0; }\n"}
SOURCES = {"python3": "solution.py", "cpp17": "solution.cpp"}
RUN_TESTS = r"""#!/bin/bash
# Checks solution.{lang} on the examples in tests/ (whitespace-token comparison, {limit}s per example).
cd "$(dirname "$0")"
{build}
fail=0
for inp in tests/example_*.in; do
  exp="${{inp%.in}}.out"
  got="$(/usr/bin/perl -e 'alarm shift; exec @ARGV' {limit} {run} < "$inp" 2>.stderr)"
  if [ $? -ne 0 ]; then echo "FAIL $inp (crash or timeout)"; tail -5 .stderr; fail=1; continue; fi
  if {py} -c 'import sys; sys.exit(sys.argv[1].split() != open(sys.argv[2]).read().split())' "$got" "$exp"; then
    echo "PASS $inp"
  else
    echo "FAIL $inp"; echo "expected:"; head -c 400 "$exp"; echo; echo "got:"; echo "$got" | head -c 400; echo
    fail=1
  fi
done
exit $fail
"""
BUILD_CPP = ('mkdir -p .build && {cpp} -std=c++17 -O2 -I .include solution.cpp -o .build/solution '
             '|| {{ echo "FAIL compile"; exit 1; }}')


@dataclass(frozen=True)
class Case:
    root: Path
    work: Path
    home: Path
    solution: Path


def pi_models_json(port: int) -> dict:
    model = {"id": "default_model", "contextWindow": CONTEXT_WINDOW, "maxTokens": MAX_TOKENS}
    return {"providers": {"local": {"baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions",
                                    "apiKey": "dummy", "models": [model]}}}


def run_tests_script(lang: str, limit: float) -> str:
    if lang == "cpp17":
        build, run = BUILD_CPP.format(cpp=cc.CPP), ".build/solution"
    else:
        build, run = "", f"{cc.PY3} solution.py"
    return RUN_TESTS.format(lang=cc.LANGS[lang]["ext"][1:], limit=limit, build=build, run=run, py=cc.PY3)


def build_workspace(run_dir: Path, task: dict, lang: str, root: Path) -> Case:
    root = root.resolve()
    work, home = root / "work", root / "home"
    (work / "tests").mkdir(parents=True)
    (root / "tmp").mkdir()
    task_dir = (run_dir / task["tests_path"]).parent
    meta = cc.load_json(task_dir / "task.json")
    (work / "PROBLEM.md").write_text((run_dir / task["prompts"][lang]["path"]).read_text(encoding="utf-8"))
    solution = work / SOURCES[lang]
    solution.write_text(STUBS[lang])
    rows = [json.loads(line) for line in (run_dir / task["tests_path"]).read_text().splitlines()]
    for k, row in enumerate(r for r in rows if r["source"] == "public_tests"):
        (work / f"tests/example_{k}.in").write_text(row["input"])
        (work / f"tests/example_{k}.out").write_text(row["expected"])
    if lang == "cpp17":
        shutil.copytree(cc.write_bits_compat(root / "tmp"), work / ".include")
    script = work / "run_tests.sh"
    script.write_text(run_tests_script(lang, float(meta["time_limit_seconds"]) * EXAMPLE_TIME_FACTOR))
    script.chmod(0o755)
    (home / ".pi/agent").mkdir(parents=True)
    (home / ".pi/agent/models.json").write_text(json.dumps(pi_models_json(PORT), indent=2))
    return Case(root=root, work=work, home=home, solution=solution)


def task_message(lang: str) -> str:
    label = cc.LANGS[lang]["label"]
    return (f"Solve the programming problem in PROBLEM.md. Write the complete program in {label} to "
            f"{SOURCES[lang]} in this directory; it must read standard input and write standard output. "
            "The examples are in tests/; run ./run_tests.sh to check them. Hidden tests will be run on "
            f"{SOURCES[lang]} afterwards, so make it correct and efficient. Stop when you are done.")


def pi_sandbox_profile(case_root: Path, port: int) -> str:
    home = Path.home()
    readable = sorted({home / ".local", NODE.resolve().parents[1]})
    allow_reads = " ".join(f'(require-not (subpath "{p}"))' for p in readable)
    root = case_root.resolve()
    return (
        "(version 1)\n(allow default)\n"
        "(deny network*)\n"
        f'(allow network-outbound (remote ip "localhost:{port}"))\n'
        "(allow network* (remote unix-socket))\n"
        f'(deny file-read-data (require-all (subpath "{home}") {allow_reads}))\n'
        "(deny file-write*)\n"
        f'(allow file-write* (subpath "{root}") (literal "/dev/null") (literal "/dev/tty")'
        ' (regex #"^/dev/fd/") (regex #"^/dev/ttys")'
        ' (regex #"^/private/var/folders/.*/xcrun_db"))\n'  # clang's xcrun cache, else a warning per build
    )


def parse_events(lines: Iterable[str]) -> dict:
    stats = {"model_calls": 0, "tool_calls": 0, "tokens_in": 0, "tokens_out": 0, "cache_read": 0}
    for line in lines:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = event.get("message") if isinstance(event, dict) else None
        if event.get("type") != "message_end" or not message or message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        stats["model_calls"] += 1
        stats["tool_calls"] += sum(1 for c in message.get("content", []) if c.get("type") == "toolCall")
        stats["tokens_in"] += int(usage.get("input", 0))
        stats["tokens_out"] += int(usage.get("output", 0))
        stats["cache_read"] += int(usage.get("cacheRead", 0))
    return stats


def paired_summary(cases: list[dict], a: str, b: str) -> dict:
    problems = sorted({c["task_id"] for c in cases})
    by_problem = {p: [float(c[a]) - float(c[b]) for c in cases if c["task_id"] == p] for p in problems}
    rng = random.Random(cc.SEED)
    boot = []
    for _ in range(cc.BOOTSTRAPS):
        values = [d for p in (rng.choice(problems) for _ in problems) for d in by_problem[p]]
        boot.append(sum(values) / len(values))
    boot.sort()
    n = len(cases)
    a_passed, b_passed = sum(bool(c[a]) for c in cases), sum(bool(c[b]) for c in cases)
    return {"n_cases": n, f"{a}_passed": a_passed, f"{b}_passed": b_passed,
            f"{a}_only": sum(bool(c[a]) and not c[b] for c in cases),
            f"{b}_only": sum(bool(c[b]) and not c[a] for c in cases),
            "delta": (a_passed - b_passed) / n,
            "bootstrap_95": [cc.percentile(boot, 0.025), cc.percentile(boot, 0.975)],
            "bootstrap_samples": cc.BOOTSTRAPS, "bootstrap_seed": cc.SEED}


def pi_env(case: Case) -> dict[str, str]:
    return {"HOME": str(case.home), "TMPDIR": str(case.root / "tmp"), "PI_OFFLINE": "1", "PI_TELEMETRY": "0",
            "PATH": f"{NODE.parent}:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin", "LANG": "en_US.UTF-8"}


def kill_group(proc: subprocess.Popen) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=KILL_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def watch_pi(proc: subprocess.Popen, events_path: Path, start: float) -> str:
    """Stream pi's stdout to events_path; enforce the wall-time and tool-call limits."""
    tool_calls = 0
    os.set_blocking(proc.stdout.fileno(), False)
    buffer = b""
    with events_path.open("wb") as out:
        while True:
            chunk = proc.stdout.read() or b""
            out.write(chunk)
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            tool_calls += parse_events(line.decode("utf-8", "replace") for line in lines)["tool_calls"]
            if tool_calls > TOOL_CALL_LIMIT:
                kill_group(proc)
                return "tool_limit"
            if time.monotonic() - start > WALL_LIMIT_S:
                kill_group(proc)
                return "timeout"
            if proc.poll() is not None and not chunk:
                out.write(proc.stdout.read() or b"")
                return "done" if proc.returncode == 0 else f"error:{proc.returncode}"
            time.sleep(0.2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_pi(case: Case, lang: str, case_out: Path) -> dict[str, Any]:
    profile = pi_sandbox_profile(case.root, PORT)
    cmd = ["/usr/bin/sandbox-exec", "-p", profile, str(NODE), str(PI.resolve()), "--offline",
           "--provider", "local", "--model", "default_model", "--thinking", "off", "--mode", "json",
           "--print", task_message(lang)]
    start = time.monotonic()
    with (case_out / "pi.err").open("wb") as err:
        proc = subprocess.Popen(cmd, cwd=case.work, env=pi_env(case), stdout=subprocess.PIPE, stderr=err,
                                stdin=subprocess.DEVNULL, start_new_session=True)
        try:
            reason = watch_pi(proc, case_out / "events.jsonl", start)
        finally:
            kill_group(proc)
    return {"exit_reason": reason, "wall_s": round(time.monotonic() - start, 1)}


def run_case(run_dir: Path, task: dict, lang: str, arm: str, out_dir: Path) -> dict:
    case_out = out_dir / arm / f"{task['task_id']}-{lang}"
    if (case_out / "record.json").exists():
        return cc.load_json(case_out / "record.json")
    case_out.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix=f"pi-eval-{arm}-{task['task_id']}-{lang}-", dir="/private/tmp"))
    try:
        case = build_workspace(run_dir, task, lang, root)
        record = {"arm": arm, "task_id": task["task_id"], "language": lang, "started_utc": cc.now(),
                  "memory_before": cc.memory_free()}
        record.update(run_pi(case, lang, case_out))
        record.update(parse_events((case_out / "events.jsonl").read_text(errors="replace").splitlines()))
        record.update({"finished_utc": cc.now(), "memory_after": cc.memory_free(),
                       "solution_sha256": sha256_file(case.solution)})
        dest = cc.result_dir(run_dir, task, f"agent-{arm}", lang)
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(case.solution, dest / f"solution{cc.LANGS[lang]['ext']}")
        cc.write_json(dest / "generation.json", {"status": "complete", "model": f"agent-{arm}",
                                                 "task_id": task["task_id"], "language": lang,
                                                 "record": str(case_out / "record.json")})
        cc.write_json(case_out / "record.json", record)
        return record
    finally:
        shutil.rmtree(root, ignore_errors=True)


def server_args(arm: str) -> list[str]:
    args = [str(cc.SERVER), "--model", str(ARM_MODELS[arm]), "--host", "127.0.0.1", "--port", str(PORT),
            "--temp", str(TEMPERATURE), "--top-p", str(TOP_P), "--chat-template-args",
            json.dumps({"enable_thinking": False}), "--prompt-concurrency", "1", "--decode-concurrency", "1",
            "--prompt-cache-bytes", str(PROMPT_CACHE_BYTES),
            "--prefill-step-size", str(PREFILL_STEP_SIZE)]
    if arm == "pi":
        if not (ADAPTER / "adapters.safetensors").exists():
            raise FileNotFoundError(f"No converted adapter at {ADAPTER} (run tools/peft_to_mlx.py)")
        args += ["--adapter-path", str(ADAPTER)]
    return args


def server_errors(log_path: Path, offset: int) -> list[str]:
    """Fatal server log lines written after byte `offset` (a dead generation thread, then 404s)."""
    with log_path.open("rb") as f:
        f.seek(offset)
        new = f.read().decode(errors="replace")
    return [line for line in new.splitlines() if any(m in line for m in SERVER_FATAL)]


def assert_server_healthy(proc: subprocess.Popen, log_path: Path, offset: int, case_out: Path) -> None:
    """After a case: if the server died or logged a fatal error, invalidate that case and stop the run."""
    errors = server_errors(log_path, offset)
    if proc.poll() is None and not errors:
        return
    record = case_out / "record.json"
    if record.exists():
        record.rename(case_out / "record.invalid.json")
    raise RuntimeError(f"mlx_lm.server unhealthy during {case_out.name} (exit {proc.poll()}): "
                       f"{errors[:2]}; case invalidated, see {log_path}")


def cases_in_order(manifest: dict) -> list[tuple[dict, str]]:
    return [(task, lang) for task in manifest["task_selection"]["tasks"] for lang in cc.LANGS]


def run(run_dir: Path, arm: str, out_dir: Path, limit: int | None) -> None:
    cc.assert_no_server()
    if not PI.exists():
        raise FileNotFoundError(f"pi not installed at {PI}")
    manifest = cc.load_json(run_dir / "manifest.json")
    (out_dir / arm).mkdir(parents=True, exist_ok=True)
    todo = [(t, l) for t, l in cases_in_order(manifest)
            if not (out_dir / arm / f"{t['task_id']}-{l}" / "record.json").exists()]
    todo = todo[:limit] if limit is not None else todo
    log_path = out_dir / arm / "server.log"
    with log_path.open("ab") as log:
        proc = subprocess.Popen(server_args(arm), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            cc.wait_server(proc)
            for i, (task, lang) in enumerate(todo, 1):
                offset = log_path.stat().st_size
                rec = run_case(run_dir, task, lang, arm, out_dir)
                assert_server_healthy(proc, log_path, offset, out_dir / arm / f"{task['task_id']}-{lang}")
                print(f"[{arm}] {i}/{len(todo)} {task['task_id']} {lang}: {rec['exit_reason']} "
                      f"{rec['wall_s']}s {rec['tool_calls']} tools {rec['tokens_out']} out", flush=True)
        finally:
            cc.stop_server(proc)
    score_recorded(run_dir, manifest, arm, out_dir)


def score_recorded(run_dir: Path, manifest: dict, arm: str, out_dir: Path) -> None:
    """Score only the cases that have a record (cc.score_model would also score unrun cases as no_source)."""
    recorded = {(t["task_id"], l) for t, l in cases_in_order(manifest)
                if (out_dir / arm / f"{t['task_id']}-{l}" / "record.json").exists()}
    tasks = [t for t in manifest["task_selection"]["tasks"] if any(t["task_id"] == r[0] for r in recorded)]
    for task in tasks:
        for lang in cc.LANGS:
            dest = cc.result_dir(run_dir, task, f"agent-{arm}", lang)
            if (task["task_id"], lang) not in recorded or (dest / "summary.json").exists():
                continue
            s = cc.score_one(run_dir, task, f"agent-{arm}", lang)
            print(f"[{arm}] {task['task_id']} {lang}: {s['passed_test_inputs']}/{s['total_test_inputs']} "
                  f"{'PASS' if s['task_pass'] else 'FAIL'}", flush=True)


def arm_stats(rows: list[dict], arm: str) -> dict:
    return {"passed": sum(r[arm] for r in rows), "n": len(rows),
            "median_wall_s": statistics.median(r[f"{arm}_record"]["wall_s"] for r in rows),
            "median_tool_calls": statistics.median(r[f"{arm}_record"]["tool_calls"] for r in rows),
            "tokens_in": sum(r[f"{arm}_record"]["tokens_in"] for r in rows),
            "tokens_out": sum(r[f"{arm}_record"]["tokens_out"] for r in rows),
            "exit_reasons": sorted({r[f"{arm}_record"]["exit_reason"] for r in rows})}


def collect_rows(run_dir: Path, out_dir: Path) -> list[dict]:
    """One row per case with pass + record for every arm that has a scored record."""
    manifest = cc.load_json(run_dir / "manifest.json")
    rows = []
    for task, lang in cases_in_order(manifest):
        row: dict[str, Any] = {"task_id": task["task_id"], "tier": task["tier"], "language": lang}
        for arm in ARMS:
            rec_path = out_dir / arm / f"{task['task_id']}-{lang}" / "record.json"
            score_path = cc.result_dir(run_dir, task, f"agent-{arm}", lang) / "summary.json"
            if rec_path.exists() and score_path.exists():
                row[arm] = bool(cc.load_json(score_path)["task_pass"])
                row[f"{arm}_record"] = cc.load_json(rec_path)
        rows.append(row)
    return rows


def summarize_pairs(rows: list[dict], pairs: Iterable[tuple[str, str]]) -> dict:
    """Paired comparison per (a, b) over the cases both arms ran; pairs with no shared case are left out."""
    out = {}
    for a, b in pairs:
        shared = [r for r in rows if a in r and b in r]
        if shared:
            out[f"{a}_vs_{b}"] = {"paired": paired_summary(shared, a, b),
                                  "arms": {arm: arm_stats(shared, arm) for arm in (a, b)}}
    return out


def summary(run_dir: Path, out_dir: Path, dest: Path | None = None) -> dict:
    pairs = summarize_pairs(collect_rows(run_dir, out_dir), PAIRS)
    if not pairs:
        raise SystemExit("No case has scored records for two arms yet.")
    rows = [r for r in collect_rows(run_dir, out_dir) if any(arm in r for arm in ARMS)]
    result = {"created_utc": cc.now(), "pairs": pairs, "cases": rows}
    cc.write_json(dest or out_dir / "eval-b.json", result)
    for name, pair in pairs.items():
        p, (a, b) = pair["paired"], name.split("_vs_")
        print(f"{a} {p[f'{a}_passed']}/{p['n_cases']} vs {b} {p[f'{b}_passed']}/{p['n_cases']}: "
              f"{p['delta'] * 100:+.1f} pp, 95% CI [{p['bootstrap_95'][0] * 100:+.1f}, "
              f"{p['bootstrap_95'][1] * 100:+.1f}]", flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = parser.add_subparsers(dest="command", required=True)
    r = subs.add_parser("run", help="run one arm (resumable)")
    r.add_argument("--arm", choices=ARMS, required=True)
    r.add_argument("--limit", type=int, default=None, help="at most N new cases")
    s = subs.add_parser("summary", help="pair the arms and write eval-b.json (or --dest)")
    s.add_argument("--dest", type=Path, default=None, help="output JSON (default OUT/eval-b.json)")
    for p in (r, s):
        p.add_argument("--out", type=Path, required=True)
        p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args()
    if args.command == "run":
        run(args.run_dir.resolve(), args.arm, args.out.resolve(), args.limit)
    else:
        summary(args.run_dir.resolve(), args.out.resolve(), args.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
