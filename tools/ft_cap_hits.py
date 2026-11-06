#!/usr/bin/env python3
"""Measure completion-cap hits while running the frozen heldout-v2 evaluator.

Call-level rates use model calls as their denominator. Per-task cap fractions
and the pass/cap 2x2 use task-sample runs, where a run is capped if any call
made for it returned finish_reason="length".
"""
import argparse
import io
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO

try:
    import ft_eval
except ModuleNotFoundError:
    from tools import ft_eval


WINDOW_CHARS = 50
REPETITION_THRESHOLD = 0.5
P90_FRACTION = 0.9
TOKEN_LIMIT_REASON = "length"
PLAN_REQUESTS_PER_ATTEMPT = 2
STANDARD_REQUESTS_PER_ATTEMPT = 1


def repetition_score(content: str, window_chars: int = WINDOW_CHARS) -> float:
    """Return the share of overlapping fixed-width windows seen earlier."""
    if window_chars <= 0:
        raise ValueError("window_chars must be positive")
    window_count = max(0, len(content) - window_chars + 1)
    if not window_count:
        return 0.0
    seen: set[str] = set()
    repeats = 0
    for start in range(window_count):
        window = content[start:start + window_chars]
        repeats += int(window in seen)
        seen.add(window)
    return repeats / window_count


def percentile_nearest_rank(values: list[int], fraction: float) -> int | None:
    """Return a nearest-rank percentile, or None when there are no values."""
    if not values:
        return None
    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def _call_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["task_id"]), int(row["sample_index"])


def summarize_calls(rows: list[dict[str, Any]], outcomes: dict[tuple[str, int], bool]) -> dict[str, Any]:
    """Summarize call metrics and task-sample cap/pass counts."""
    n_calls = len(rows)
    capped_calls = [row for row in rows if row.get("finish_reason") == TOKEN_LIMIT_REASON]
    token_counts = [row["completion_tokens"] for row in rows if row.get("completion_tokens") is not None]
    repetition_values = [row["repetition_score"] for row in rows if row.get("repetition_score") is not None]
    calls_by_run: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        calls_by_run[_call_key(row)].append(row)
    task_runs: dict[str, list[bool]] = defaultdict(list)
    cells = {passed: {cap: 0 for cap in (True, False)} for passed in (True, False)}
    for run_key, run_rows in calls_by_run.items():
        run_capped = any(row.get("finish_reason") == TOKEN_LIMIT_REASON for row in run_rows)
        task_runs[run_key[0]].append(run_capped)
        if run_key in outcomes:
            cells[bool(outcomes[run_key])][run_capped] += 1
    return {
        "n_calls": n_calls,
        "cap_hit_rate": len(capped_calls) / n_calls if n_calls else None,
        "mean_completion_tokens": statistics.mean(token_counts) if token_counts else None,
        "median_completion_tokens": statistics.median(token_counts) if token_counts else None,
        "p90_completion_tokens": percentile_nearest_rank(token_counts, P90_FRACTION),
        "repetition_gt_0_5_share": (
            sum(score > REPETITION_THRESHOLD for score in repetition_values) / len(repetition_values)
            if repetition_values else None
        ),
        "per_task_cap_hit_fraction": {
            task_id: sum(task_caps) / len(task_caps) for task_id, task_caps in sorted(task_runs.items())
        },
        "passed_vs_cap_hit_2x2": {
            "passed": {"cap_hit": cells[True][True], "no_cap_hit": cells[True][False]},
            "failed": {"cap_hit": cells[False][True], "no_cap_hit": cells[False][False]},
        },
    }


def _task_id(task: Any, args: Any) -> str:
    if args.split in {"public", "heldout-v2"}:
        return str(task["id"])
    return str(task[0])


class CallCapture:
    """Track evaluator context and append one metadata row per HTTP response."""

    def __init__(self, output: TextIO):
        self.output = output
        self.rows: list[dict[str, Any]] = []
        self.outcomes: dict[tuple[str, int], bool] = {}
        self.task_id: str | None = None
        self.sample_index = 0
        self.attempt = 1
        self.request_index = 0

    def begin_sample(self, task: Any, args: Any, sample_index: int) -> None:
        self.task_id = _task_id(task, args)
        self.sample_index = sample_index
        self.attempt = 1
        self.request_index = 0

    def begin_request(self, args: Any) -> None:
        requests_per_attempt = PLAN_REQUESTS_PER_ATTEMPT if args.plan_first else STANDARD_REQUESTS_PER_ATTEMPT
        self.attempt = self.request_index // requests_per_attempt + 1
        self.request_index += 1

    def record_response(self, response: Any) -> None:
        choices = response.get("choices", []) if isinstance(response, dict) else []
        choice = choices[0] if choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = message.get("content") or "" if isinstance(message, dict) else ""
        score = repetition_score(content) if isinstance(content, str) else 0.0
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        row = {
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "attempt": self.attempt,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "completion_tokens": usage.get("completion_tokens") if isinstance(usage, dict) else None,
            "repetition_score": score,
        }
        self.rows.append(row)
        self.output.write(json.dumps(row, sort_keys=True) + "\n")
        self.output.flush()

    def request_model(self, original: Any, *args: Any, **kwargs: Any) -> Any:
        original_urlopen = ft_eval.urllib.request.urlopen

        def capture_open(request: Any, timeout: int = 3600) -> io.BytesIO:
            with original_urlopen(request, timeout=timeout) as response:
                raw_response = response.read()
            self.record_response(json.loads(raw_response))
            return io.BytesIO(raw_response)

        ft_eval.urllib.request.urlopen = capture_open
        try:
            return original(*args, **kwargs)
        finally:
            ft_eval.urllib.request.urlopen = original_urlopen


def _wrapper_arguments(argv: list[str] | None) -> tuple[Path, Path, list[str]]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in raw_args or "-h" in raw_args:
        print("Additional options: --calls-out PATH --summary-out PATH; only --split heldout-v2 is supported.")
        ft_eval.arguments(["--help"])
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--calls-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    outputs, evaluator_args = parser.parse_known_args(raw_args)
    if not any(value == "--split" or value.startswith("--split=") for value in evaluator_args):
        evaluator_args.extend(["--split", "heldout-v2"])
    parsed = ft_eval.arguments(evaluator_args)
    if parsed.split != "heldout-v2":
        raise SystemExit("ft_cap_hits measures only --split heldout-v2")
    return outputs.calls_out, outputs.summary_out, evaluator_args


def _run_evaluator(evaluator_args: list[str], capture: CallCapture) -> int:
    original_model = ft_eval.request_model
    original_request = ft_eval.request_for_task
    original_evaluate = ft_eval.evaluate_one

    def wrapped_model(*args: Any, **kwargs: Any) -> Any:
        return capture.request_model(original_model, *args, **kwargs)

    def wrapped_request(prompt: str, args: Any, sample_seed: int | None, messages: Any = None) -> Any:
        capture.begin_request(args)
        return original_request(prompt, args, sample_seed, messages)

    def wrapped_evaluate(task: Any, args: Any, sample_index: int = 0) -> Any:
        capture.begin_sample(task, args, sample_index)
        result = original_evaluate(task, args, sample_index)
        capture.outcomes[(capture.task_id or "", sample_index)] = bool(result["passed"])
        return result

    ft_eval.request_model = wrapped_model
    ft_eval.request_for_task = wrapped_request
    ft_eval.evaluate_one = wrapped_evaluate
    try:
        return ft_eval.main(evaluator_args)
    finally:
        ft_eval.request_model = original_model
        ft_eval.request_for_task = original_request
        ft_eval.evaluate_one = original_evaluate


def main(argv: list[str] | None = None) -> int:
    calls_path, summary_path, evaluator_args = _wrapper_arguments(argv)
    calls_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with calls_path.open("w", encoding="utf-8") as calls_file:
        capture = CallCapture(calls_file)
        try:
            result = _run_evaluator(evaluator_args, capture)
        finally:
            summary = summarize_calls(capture.rows, capture.outcomes)
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
