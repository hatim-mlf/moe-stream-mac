<task>
Measure how often our local model's answers run to the token cap (loops / runaway output) on the frozen heldout-v2
eval, without changing tools/ft_eval.py's request body or scoring. This decides whether a "remove the repetition
direction" weight edit (MIT weight-space programming paper; see docs/research/weight-surgery-claim.md) is worth
trying on Qwen3.6-35B-A3B: only if cap hits are common.

Write `tools/ft_cap_hits.py`: runs ft_eval.main(argv) in-process with `ft_eval.request_model` wrapped (read
tools/ft_eval.py:96 onward; it currently returns only message content). The wrapper sends the identical request
body, but keeps the full JSON response and records per call: task_id (thread it through by wrapping the caller at
ft_eval.py:130-137 or by recording the prompt hash and mapping it back to task ids from the suite), sample index,
attempt, finish_reason, usage.completion_tokens, and a cheap repetition score (fraction of the content's 50-char
windows that repeat an earlier window; constant). It must return the same content to ft_eval so results are
unchanged. CLI: all ft_eval flags pass through, plus `--calls-out PATH` (jsonl, one row per model call) and
`--summary-out PATH` (json: n_calls, cap_hit_rate = finish_reason == "length", mean/median/p90 completion_tokens,
share with repetition score > 0.5, per-task cap-hit fraction, and passed-vs-cap-hit 2x2 counts).
Tests `tools/test_ft_cap_hits.py` (pytest, no network): stub the HTTP layer; assert ft_eval receives identical
content and request bodies, rows are recorded with correct task ids/attempts, summary math on a fixed set of rows,
repetition score on known strings.
</task>
<constraints>
Do not modify tools/ft_eval.py or anything under evals/. Stdlib only. Named constants, type hints, functions < 50
lines, no silent excepts. You cannot reach localhost; the lead runs it on the Mac. Do not commit.
Gate: `.venv/bin/python -m pytest -q tools/test_ft_cap_hits.py` (or python3 -m pytest if the venv is unavailable).
Report: files, gate output, how task ids are attached, anything unverified.
</constraints>
