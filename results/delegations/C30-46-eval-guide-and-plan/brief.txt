# Brief 46: optional coding guide and plan-first mode in ft_eval (Codex)

<context>
`tools/ft_eval.py` scores our local model on the frozen heldout-v2 suite. We want to measure inference-time
methods (no training): (a) a short coding checklist derived from the model's own failures
(`docs/finetune/CODING_CHECKLIST.md`), (b) asking for a plan before the code, combined with the existing
test-feedback repair loop (`--attempts N`). Read ft_eval.py fully first (`request_model`, SYSTEM, how the task
prompt is built, the attempts loop, the result JSON, `--paired`).
</context>

<task>
Add two optional flags to `tools/ft_eval.py`; with neither flag, requests, results and the paired check must be
byte-for-byte what they are today:
1. `--guide FILE`: append the file's text to the system prompt for every request (first attempt and repairs),
   separated by a blank line. Record `"guide": "<path>"` and `"guide_sha256"` in the result JSON.
2. `--plan-first`: two calls per attempt. Call 1 sends the normal prompt plus "First, do not write code. List
   the spec's inputs, outputs, error cases and edge cases, then a short plan, in at most 15 lines." Call 2 sends
   the conversation (prompt, the plan as the assistant turn) plus "Now write the solution as one fenced code
   block." Only call 2's reply is extracted and tested. In repair attempts, keep today's feedback format, with
   the plan turn kept in the history. Record `"plan_first": true` and store each plan (≤ 2,000 chars) in the
   per-task result.
3. `--paired` must refuse to pair files whose guide sha or plan_first differ from each other's unless
   `--allow-method-diff` is given (we pair method A vs base on purpose; the flag makes it explicit). It must
   still refuse when suite, temperature or sample count differ.
</task>

<tests>
Extend `tools/test_ft_eval_v2.py` (or add tools/test_ft_eval_methods.py) with a fake endpoint or a monkeypatched
`request_model`: without flags the request bodies are unchanged (compare against a captured baseline);
`--guide` appears in the system prompt of every call; `--plan-first` makes two calls in the right order and only
the second reply is tested; result JSON fields; paired refusal and `--allow-method-diff`.
Run `.venv/bin/python -m pytest -q tools/test_ft_eval*.py`.
</tests>

<constraints>
Do not touch evals/ or the frozen suite hashes. Named constants, functions < 50 lines, no silent excepts. No
commit. Report files, test output, and the exact text of both plan-first instructions.
</constraints>
