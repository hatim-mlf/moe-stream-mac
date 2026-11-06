"""Offline tests for cap-hit instrumentation around the frozen evaluator."""
import contextlib
import io
import json
from pathlib import Path
from typing import Any
from unittest import mock

import ft_cap_hits


def test_repetition_score_uses_overlapping_windows() -> None:
    assert ft_cap_hits.repetition_score("short") == 0.0
    assert ft_cap_hits.repetition_score("a" * 60) == 10 / 11
    assert ft_cap_hits.repetition_score("A" * 50 + "B") == 0.0


def test_summary_metrics_and_pass_cap_matrix() -> None:
    rows = [
        {"task_id": "a", "sample_index": 0, "attempt": 1, "finish_reason": "length",
         "completion_tokens": 100, "repetition_score": 0.6},
        {"task_id": "a", "sample_index": 0, "attempt": 2, "finish_reason": "stop",
         "completion_tokens": 1000, "repetition_score": 0.8},
        {"task_id": "a", "sample_index": 1, "attempt": 1, "finish_reason": "stop",
         "completion_tokens": 200, "repetition_score": 0.4},
        {"task_id": "b", "sample_index": 0, "attempt": 1, "finish_reason": "length",
         "completion_tokens": 300, "repetition_score": 0.7},
        {"task_id": "b", "sample_index": 1, "attempt": 1, "finish_reason": "stop",
         "completion_tokens": 400, "repetition_score": 0.2},
    ]
    outcomes = {("a", 0): True, ("a", 1): True, ("b", 0): False, ("b", 1): False}

    summary = ft_cap_hits.summarize_calls(rows, outcomes)

    assert summary["n_calls"] == 5
    assert summary["cap_hit_rate"] == 2 / 5
    assert summary["mean_completion_tokens"] == 400
    assert summary["median_completion_tokens"] == 300
    assert summary["p90_completion_tokens"] == 1000
    assert summary["repetition_gt_0_5_share"] == 3 / 5
    assert summary["per_task_cap_hit_fraction"] == {"a": 0.5, "b": 0.5}
    assert summary["passed_vs_cap_hit_2x2"] == {
        "passed": {"cap_hit": 1, "no_cap_hit": 1},
        "failed": {"cap_hit": 1, "no_cap_hit": 1},
    }


def _expected_body(prompt: str) -> dict[str, Any]:
    return {
        "model": "local", "temperature": 0.0, "max_tokens": 8192,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "system", "content": ft_cap_hits.ft_eval.SYSTEM},
                     {"role": "user", "content": prompt}],
    }


def _run_stubbed_eval(
    tmp_path: Path,
) -> tuple[int, list[str], list[str], list[bytes], Path, Path, Path]:
    task = {"id": "case-a", "target": "answer.py", "build_cmd": [], "test_cmd": [],
            "timeout_s": 1, "_folder": str(tmp_path), "_prompt": "frozen prompt"}
    replies = [
        {"choices": [{"message": {"content": "```python\nfirst\n```"},
                       "finish_reason": "length"}], "usage": {"completion_tokens": 8192}},
        {"choices": [{"message": {"content": "```python\nsecond\n```"},
                       "finish_reason": "stop"}], "usage": {"completion_tokens": 37}},
    ]
    requests = []
    tested_code = []
    received_content = []

    def fake_urlopen(request: Any, timeout: int) -> io.BytesIO:
        requests.append(request.data)
        return io.BytesIO(json.dumps(replies[len(requests) - 1]).encode())

    def fake_check(check_task: Any, code: str) -> tuple[int, str]:
        tested_code.append(code)
        return (1, "repair needed") if len(tested_code) == 1 else (0, "ok")

    original_extract = ft_cap_hits.ft_eval.extract_code

    def capture_content(content: str) -> str:
        received_content.append(content)
        return original_extract(content)

    calls_path = tmp_path / "calls.jsonl"
    summary_path = tmp_path / "summary.json"
    result_path = tmp_path / "eval.json"
    args = ["--split", "heldout-v2", "--attempts", "2", "--model", "local",
            "--endpoint", "http://127.0.0.1:8080/v1/chat/completions",
            "--out", str(result_path), "--calls-out", str(calls_path),
            "--summary-out", str(summary_path)]
    with contextlib.redirect_stdout(io.StringIO()), \
            mock.patch.object(ft_cap_hits.ft_eval, "task_records", return_value=[task]), \
            mock.patch.object(ft_cap_hits.ft_eval, "verify_v2_manifest",
                              return_value={"suite_sha256": "frozen-suite-hash"}), \
            mock.patch.object(ft_cap_hits.ft_eval, "run_v2_candidate_check", side_effect=fake_check), \
            mock.patch.object(ft_cap_hits.ft_eval, "extract_code", side_effect=capture_content), \
            mock.patch.object(ft_cap_hits.ft_eval.urllib.request, "urlopen", side_effect=fake_urlopen):
        result = ft_cap_hits.main(args)
    return result, tested_code, received_content, requests, calls_path, summary_path, result_path


def test_in_process_eval_preserves_content_and_request_body(tmp_path: Path) -> None:
    result, tested_code, received_content, requests, calls_path, summary_path, result_path = _run_stubbed_eval(tmp_path)
    assert result == 0
    assert tested_code == ["first", "second"]
    assert received_content == ["```python\nfirst\n```", "```python\nsecond\n```"]
    assert len(requests) == 2
    first_prompt = "frozen prompt"
    retry_prompt = (first_prompt + "\n\n## Attempt 1 failed\n\nYour code:\n```\nfirst\n```"
                    "\n\nTest output:\n```\nrepair needed\n```\n\nFix every reported issue; change nothing else.")
    assert requests[0] == json.dumps(_expected_body(first_prompt)).encode()
    assert requests[1] == json.dumps(_expected_body(retry_prompt)).encode()
    rows = [json.loads(line) for line in calls_path.read_text().splitlines()]
    assert [(row["task_id"], row["sample_index"], row["attempt"]) for row in rows] == [
        ("case-a", 0, 1), ("case-a", 0, 2),
    ]
    assert [row["finish_reason"] for row in rows] == ["length", "stop"]
    assert [row["completion_tokens"] for row in rows] == [8192, 37]
    assert json.loads(result_path.read_text())["results"][0]["passed"] is True
    assert json.loads(summary_path.read_text())["passed_vs_cap_hit_2x2"]["passed"]["cap_hit"] == 1
