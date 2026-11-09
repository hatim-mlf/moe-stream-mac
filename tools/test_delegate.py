"""Tests for the automatic repair loop in tools/delegate.py (no model needed: `ask` is faked)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import delegate  # noqa: E402

FENCE = "`" * 3


def reply_with(code: str) -> str:
    return f"{FENCE}python\n{code}\n{FENCE}"


class FakeModel:
    """Returns queued replies and records every prompt it was sent."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.prompts: list[str] = []

    def __call__(self, prompt: str, system: str = "") -> tuple[str, dict, float]:
        self.prompts.append(prompt)
        return self.replies.pop(0), {"completion_tokens": 10, "total_tokens": 30}, 1.5


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(delegate, "ROOT", tmp_path)
    out = tmp_path / "out.py"
    # The test command passes only when the written file defines GOOD = 1.
    test_cmd = f"{sys.executable} -c \"import runpy,sys; sys.exit(0 if runpy.run_path('{out}').get('GOOD') == 1 else 3)\""
    return tmp_path, out, test_cmd


def outcomes(root: Path) -> list[dict]:
    path = root / "results" / "delegations" / "outcomes.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_passes_first_try_uses_one_attempt(env, monkeypatch):
    root, out, test_cmd = env
    model = FakeModel([reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="t1")

    assert result["attempts_used"] == 1
    assert result["passed_first"] is True and result["passed_final"] is True
    assert len(model.prompts) == 1
    assert out.read_text() == "GOOD = 1\n"


def test_failure_then_pass_sends_code_and_test_output_back(env, monkeypatch):
    root, out, test_cmd = env
    model = FakeModel([reply_with("GOOD = 0\nprint('marker-output')"), reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("the brief", out, test_cmd=test_cmd, attempts=3, name="t2")

    assert result["attempts_used"] == 2
    assert result["passed_first"] is False and result["passed_final"] is True
    second = model.prompts[1]
    assert second.startswith("the brief")
    assert "GOOD = 0" in second            # previous code is returned
    assert "marker-output" in second       # test output is returned
    assert "exit code 3" in second
    assert out.read_text() == "GOOD = 1\n"


def test_repair_prompt_holds_only_latest_failure(env, monkeypatch):
    root, out, test_cmd = env
    model = FakeModel([reply_with("GOOD = 0  # first"), reply_with("GOOD = 0  # second"),
                       reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="t3")

    assert "# second" in model.prompts[2] and "# first" not in model.prompts[2]


def test_gives_up_after_attempt_cap_and_keeps_last_code(env, monkeypatch):
    root, out, test_cmd = env
    model = FakeModel([reply_with(f"GOOD = 0  # {i}") for i in range(3)])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="t4")

    assert result["attempts_used"] == 3 and result["passed_final"] is False
    assert out.read_text() == "GOOD = 0  # 2\n"


def test_reply_without_code_block_counts_as_failed_attempt(env, monkeypatch):
    root, out, test_cmd = env
    model = FakeModel(["sorry, no code", reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="t5")

    assert result["attempts_used"] == 2 and result["passed_final"] is True
    assert "no code block" in model.prompts[1].lower()


def test_without_test_command_makes_one_call_and_reports_untested(env, monkeypatch):
    root, out, _ = env
    model = FakeModel([reply_with("GOOD = 0")])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("brief", out, test_cmd=None, attempts=3, name="t6")

    assert result["attempts_used"] == 1
    assert result["passed_first"] is None and result["passed_final"] is None
    assert len(model.prompts) == 1


def test_appends_one_outcome_row_per_delegation(env, monkeypatch):
    root, out, test_cmd = env
    monkeypatch.setattr(delegate, "ask", FakeModel([reply_with("GOOD = 0"), reply_with("GOOD = 1")]))
    delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="first")
    monkeypatch.setattr(delegate, "ask", FakeModel([reply_with("GOOD = 1")]))
    delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="second")

    rows = outcomes(root)
    assert [r["name"] for r in rows] == ["first", "second"]
    for key in ("utc", "attempts_used", "passed_first", "passed_final", "seconds", "tokens",
                "test_cmd", "log_dir", "adapter"):
        assert key in rows[0]
    assert rows[0]["tokens"] == 60 and rows[0]["seconds"] == 3.0


def test_long_test_output_is_clipped_to_its_tail(env, monkeypatch):
    root, out, test_cmd = env
    noisy = f"GOOD = 0\nprint('x' * {delegate.MAX_TEST_OUTPUT_CHARS * 3})\nprint('TAIL-MARK')"
    model = FakeModel([reply_with(noisy), reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    delegate.delegate("brief", out, test_cmd=test_cmd, attempts=3, name="t8")

    assert "TAIL-MARK" in model.prompts[1]
    assert len(model.prompts[1]) < len(noisy) + delegate.MAX_TEST_OUTPUT_CHARS + 2000


def test_test_command_timeout_is_reported_as_failure(env, monkeypatch):
    root, out, _ = env
    monkeypatch.setattr(delegate, "TEST_TIMEOUT_SECONDS", 1)
    slow_cmd = f"{sys.executable} -c \"import time; time.sleep(5)\""
    model = FakeModel([reply_with("GOOD = 1"), reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("brief", out, test_cmd=slow_cmd, attempts=2, name="t9")

    assert result["passed_final"] is False
    assert "timed out" in model.prompts[1].lower()


def test_test_command_runs_in_callers_directory(env, monkeypatch, tmp_path):
    root, out, _ = env
    project = tmp_path / "other-project"
    project.mkdir()
    (project / "marker.txt").write_text("here")
    monkeypatch.chdir(project)
    model = FakeModel([reply_with("GOOD = 1")])
    monkeypatch.setattr(delegate, "ask", model)

    result = delegate.delegate("brief", out, test_cmd="test -f marker.txt", attempts=1, name="t10")

    assert result["passed_final"] is True
