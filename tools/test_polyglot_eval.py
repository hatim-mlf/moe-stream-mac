"""Tests for tools/polyglot_eval.py: the solve/repair loop with a fake model, resume, and summary."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import polyglot_bench as pb  # noqa: E402
import polyglot_eval as pe  # noqa: E402
from test_polyglot_bench import GOOD_PY, make_python_exercise  # noqa: E402

FENCE = "`" * 3


def reply(code: str) -> str:
    return f"two_fer.py\n{FENCE}python\n{code}{FENCE}\n"


class FakeAsk:
    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[tuple[list[dict], bool]] = []

    def __call__(self, messages: list[dict], think: bool) -> tuple[str, dict]:
        self.calls.append((messages, think))
        return self.replies.pop(0), {"seconds": 2.0, "completion_tokens": 7, "forced": False}


@pytest.fixture
def ex(tmp_path):
    make_python_exercise(tmp_path / "bench")
    return pb.load_exercise(tmp_path / "bench", "python", "two-fer")


def last_user(ask: FakeAsk, i: int) -> str:
    return ask.calls[i][0][-1]["content"]


def test_pass_first_try(ex, tmp_path):
    ask = FakeAsk([reply(GOOD_PY)])
    r = pe.solve(ex, ask, tmp_path / "w", attempts=3, think=False)
    assert r["pass_first"] is True and r["pass_final"] is True and r["attempts_used"] == 1
    assert r["lang"] == "python" and r["name"] == "two-fer"
    assert ask.calls[0][0][-1]["content"] == pb.build_prompt(ex)


def test_repair_prompt_has_current_code_and_test_output(ex, tmp_path):
    ask = FakeAsk([reply("def two_fer(name='you'):\n    return 'WRONG-MARK'\n"), reply(GOOD_PY)])
    r = pe.solve(ex, ask, tmp_path / "w", attempts=3, think=False)
    assert r["pass_first"] is False and r["pass_final"] is True and r["attempts_used"] == 2
    second = last_user(ask, 1)
    assert "WRONG-MARK" in second                      # the code that failed
    assert "One for you, one for me." in second        # pytest's assertion output
    assert len(ask.calls[1][0]) == len(ask.calls[0][0])  # no growing chat history


def test_unparsable_reply_counts_as_failed_attempt(ex, tmp_path):
    ask = FakeAsk(["no code here", reply(GOOD_PY)])
    r = pe.solve(ex, ask, tmp_path / "w", attempts=3, think=False)
    assert r["attempts_used"] == 2 and r["pass_final"] is True
    assert "no file" in last_user(ask, 1).lower()


def test_stops_at_attempt_cap(ex, tmp_path):
    ask = FakeAsk([reply("def two_fer(name='you'):\n    return 1\n")] * 3)
    r = pe.solve(ex, ask, tmp_path / "w", attempts=3, think=False)
    assert r["attempts_used"] == 3 and r["pass_final"] is False and len(ask.calls) == 3


def test_attempt_records_have_stats(ex, tmp_path):
    ask = FakeAsk([reply(GOOD_PY)])
    r = pe.solve(ex, ask, tmp_path / "w", attempts=1, think=True)
    assert ask.calls[0][1] is True
    a = r["attempts"][0]
    for key in ("attempt", "passed", "files", "test_output", "seconds", "completion_tokens", "forced", "reply"):
        assert key in a
    assert a["files"] == ["two_fer.py"]
    assert r["seconds"] == 2.0 and r["completion_tokens"] == 7


def test_result_path_and_pending(tmp_path):
    exercises = [("python", "a"), ("cpp", "b")]
    path = pe.result_path(tmp_path, "kat", "repair3", "python", "a")
    assert path == tmp_path / "kat" / "repair3" / "python-a.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    assert pe.pending(tmp_path, "kat", "repair3", exercises) == [("cpp", "b")]


def test_summarize_counts_first_and_final(tmp_path):
    rows = [("python", "a", True, True), ("python", "b", False, True), ("cpp", "c", False, False)]
    for lang, name, first, final in rows:
        p = pe.result_path(tmp_path, "rft", "repair3", lang, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"lang": lang, "name": name, "pass_first": first, "pass_final": final,
                                 "attempts_used": 1, "seconds": 10.0, "completion_tokens": 5}))
    s = pe.summarize(tmp_path)
    assert s["rft"]["repair3"]["n"] == 3
    assert s["rft"]["repair3"]["pass_first"] == 1 and s["rft"]["repair3"]["pass_final"] == 2
    assert s["rft"]["repair3"]["by_lang"]["python"] == {"n": 2, "pass_first": 1, "pass_final": 2}
    assert s["rft"]["repair3"]["seconds"] == 30.0


def test_conditions_and_models_are_declared():
    assert pe.CONDITIONS == {"repair3": {"attempts": 3, "think": False},
                             "think": {"attempts": 1, "think": True}}
    assert set(pe.MODELS) == {"rft", "kat"}
    assert pe.MODELS["rft"]["adapter"] is not None and pe.MODELS["kat"]["adapter"] is None
