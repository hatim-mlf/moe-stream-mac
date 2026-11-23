"""Tests for tools/pi_agent_eval.py (brief 51, Eval B: the 30 code-compare cases solved agentically through pi).

  .venv/bin/python -m pytest -q tools/test_pi_agent_eval.py

No model or pi needed: workspace building, the pi sandbox, event parsing and the paired summary.
"""
import json
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
import pi_agent_eval as pe  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "results/20261108T000646Z-qwen38-code-comparison"
TASK_DIR = RUN_DIR / "tasks/easy/1579_a_casimir_s_string_solitaire"
FIXTURE = Path(__file__).parent / "fixtures/pi_session_toy.jsonl"
SANDBOX = "/usr/bin/sandbox-exec"
PY3 = "/opt/homebrew/bin/python3"  # outside the denied home dir (the venv python is under it)


def task_entry() -> dict:
    manifest = json.loads((RUN_DIR / "manifest.json").read_text())
    return next(t for t in manifest["task_selection"]["tasks"] if t["task_id"] == "1579_A")


@pytest.mark.parametrize("lang,ref", [("python3", "reference.py"), ("cpp17", "reference.cpp")])
def test_workspace_has_statement_stub_and_only_public_examples(tmp_path: Path, lang: str, ref: str) -> None:
    case = pe.build_workspace(RUN_DIR, task_entry(), lang, tmp_path)
    work = case.work
    ext = ".py" if lang == "python3" else ".cpp"
    assert case.solution == work / f"solution{ext}"
    assert case.solution.exists()
    assert "Casimir" in (work / "PROBLEM.md").read_text()
    public = [json.loads(l) for l in (TASK_DIR / "tests.jsonl").read_text().splitlines()
              if json.loads(l)["source"] == "public_tests"]
    assert sorted(p.name for p in (work / "tests").iterdir()) == sorted(
        [f"example_{i}.in" for i in range(len(public))] + [f"example_{i}.out" for i in range(len(public))])
    assert (work / "tests/example_0.in").read_text() == public[0]["input"]
    # nothing in the workspace points at or copies the hidden tests
    assert not any("tests.jsonl" in p.name for p in work.rglob("*"))
    models = json.loads((case.home / ".pi/agent/models.json").read_text())
    assert models == pe.pi_models_json(pe.PORT)


@pytest.mark.parametrize("lang,ref", [("python3", "reference.py"), ("cpp17", "reference.cpp")])
def test_run_tests_script_passes_reference_and_fails_stub(tmp_path: Path, lang: str, ref: str) -> None:
    case = pe.build_workspace(RUN_DIR, task_entry(), lang, tmp_path)
    stub = subprocess.run(["./run_tests.sh"], cwd=case.work, capture_output=True, text=True, timeout=120)
    assert stub.returncode != 0
    shutil.copy(TASK_DIR / "references" / ref, case.solution)
    good = subprocess.run(["./run_tests.sh"], cwd=case.work, capture_output=True, text=True, timeout=120)
    assert good.returncode == 0, good.stdout + good.stderr


def test_task_message_names_solution_file_and_test_script() -> None:
    for lang, name in (("python3", "solution.py"), ("cpp17", "solution.cpp")):
        msg = pe.task_message(lang)
        assert name in msg and "PROBLEM.md" in msg and "run_tests.sh" in msg


def test_models_json_points_pi_at_local_mlx_server() -> None:
    cfg = pe.pi_models_json(8080)
    provider = cfg["providers"]["local"]
    assert provider["baseUrl"] == "http://127.0.0.1:8080/v1"
    assert provider["api"] == "openai-completions"  # the card: Responses API was never trained/evaluated
    assert [m["id"] for m in provider["models"]] == ["default_model"]  # mlx_lm.server maps only this name


def test_parse_events_counts_tools_calls_and_tokens_once() -> None:
    stats = pe.parse_events(FIXTURE.read_text().splitlines())
    assert stats == {"model_calls": 4, "tool_calls": 4, "tokens_in": 1759, "tokens_out": 305, "cache_read": 5329}


def test_parse_events_ignores_garbage_lines() -> None:
    lines = ["not json", "", *FIXTURE.read_text().splitlines()]
    assert pe.parse_events(lines)["tool_calls"] == 4


def serve_once(port_holder: list) -> socket.socket:
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(4)
    port_holder.append(srv.getsockname()[1])
    threading.Thread(target=lambda: [srv.accept()[0].close() for _ in range(4)], daemon=True).start()
    return srv


def sandboxed(profile: str, code: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([SANDBOX, "-p", profile, PY3, "-I", "-c", code], cwd=cwd,
                          capture_output=True, text=True, timeout=60)


def test_sandbox_blocks_hidden_tests_outside_writes_and_other_ports(tmp_path: Path) -> None:
    case = pe.build_workspace(RUN_DIR, task_entry(), "python3", tmp_path)
    allowed, other = [], []
    s1, s2 = serve_once(allowed), serve_once(other)
    profile = pe.pi_sandbox_profile(case.root, allowed[0])

    read_ws = sandboxed(profile, "print(open('PROBLEM.md').read()[:10])", case.work)
    assert read_ws.returncode == 0, read_ws.stderr
    write_ws = sandboxed(profile, "open('scratch.txt','w').write('x')", case.work)
    assert write_ws.returncode == 0, write_ws.stderr

    hidden = sandboxed(profile, f"open({str(TASK_DIR / 'tests.jsonl')!r}).read()", case.work)
    assert hidden.returncode != 0
    repo = sandboxed(profile, f"open({str(ROOT / 'CLAUDE.md')!r}).read()", case.work)
    assert repo.returncode != 0
    outside = sandboxed(profile, f"open({str(ROOT / 'pi-escape.txt')!r},'w').write('x')", case.work)
    assert outside.returncode != 0 and not (ROOT / "pi-escape.txt").exists()

    ok = sandboxed(profile, f"import socket; socket.create_connection(('127.0.0.1',{allowed[0]}),5)", case.work)
    assert ok.returncode == 0, ok.stderr
    blocked = sandboxed(profile, f"import socket; socket.create_connection(('127.0.0.1',{other[0]}),5)", case.work)
    assert blocked.returncode != 0
    s1.close()
    s2.close()


def test_paired_summary_counts_and_clusters_by_problem() -> None:
    cases = [{"task_id": t, "language": l, "a": a, "b": b}
             for t, l, a, b in [("P1", "python3", True, False), ("P1", "cpp17", True, True),
                                ("P2", "python3", False, False), ("P2", "cpp17", True, False),
                                ("P3", "python3", False, True), ("P3", "cpp17", False, False)]]
    s = pe.paired_summary(cases, "a", "b")
    assert s["n_cases"] == 6 and s["a_passed"] == 3 and s["b_passed"] == 2
    assert s["a_only"] == 2 and s["b_only"] == 1
    assert s["delta"] == pytest.approx(1 / 6)
    lo, hi = s["bootstrap_95"]
    assert lo <= s["delta"] <= hi
    assert pe.paired_summary(cases, "a", "b") == s  # seeded, deterministic


def test_server_errors_reports_only_new_fatal_lines(tmp_path: Path) -> None:
    log = tmp_path / "server.log"
    log.write_text("2026 - ERROR - old failure from an earlier run\n")
    offset = log.stat().st_size
    assert pe.server_errors(log, offset) == []
    with log.open("a") as f:
        f.write("INFO - Prompt processing progress: 1/2\n"
                "2026 - ERROR - mlx_lm.server generation thread died: [METAL] Insufficient Memory\n"
                '127.0.0.1 - - [x] "POST /v1/chat/completions HTTP/1.1" 404 -\n')
    errors = pe.server_errors(log, offset)
    assert len(errors) == 2 and "generation thread died" in errors[0] and "404" in errors[1]


def test_server_args_cap_prompt_cache_for_both_arms() -> None:
    for arm in pe.ARMS:
        args = pe.server_args(arm)
        assert args[args.index("--prompt-cache-bytes") + 1] == str(pe.PROMPT_CACHE_BYTES)
        assert args[args.index("--prefill-step-size") + 1] == str(pe.PREFILL_STEP_SIZE)


def test_kat_arm_serves_kat_without_adapter() -> None:
    args = pe.server_args("kat")
    assert args[args.index("--model") + 1] == str(pe.KAT_MODEL)
    assert "--adapter-path" not in args
    assert pe.server_args("base")[pe.server_args("base").index("--model") + 1] == str(pe.ARM_MODELS["base"])


def _row(task: str, lang: str, **passes: bool) -> dict:
    row = {"task_id": task, "language": lang}
    for arm, ok in passes.items():
        row[arm] = ok
        row[f"{arm}_record"] = {"wall_s": 10.0, "tool_calls": 3, "tokens_in": 5, "tokens_out": 7,
                                "exit_reason": "done"}
    return row


def test_summarize_pairs_uses_only_cases_both_arms_ran() -> None:
    rows = [_row("P1", "python3", pi=True, base=False, kat=True),
            _row("P1", "cpp17", pi=True, base=True, kat=False),
            _row("P2", "python3", pi=False, base=False)]  # kat not run yet
    out = pe.summarize_pairs(rows, (("pi", "base"), ("kat", "base")))
    assert out["pi_vs_base"]["paired"]["n_cases"] == 3
    assert out["kat_vs_base"]["paired"]["n_cases"] == 2
    assert out["kat_vs_base"]["paired"]["kat_passed"] == 1
    assert out["kat_vs_base"]["arms"]["kat"]["passed"] == 1
