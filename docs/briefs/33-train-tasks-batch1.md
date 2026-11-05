<task>
Phase 4 training data, batch 1 of 5: write 50 NEW test-verified coding tasks that will be used as supervised
fine-tuning data for our local Qwen3.6-35B-A3B. You are the teacher: each task has a frozen spec, tests, a
reference solution that passes them, and at least one plausible wrong solution (mutant) that fails them.
Read first: CLAUDE.md, docs/finetune/MASTER_PLAN.md (Phase 4), docs/finetune/DIAGNOSIS.md (what the student gets
wrong: spec drift, invented semantics, ignored review notes, API misuse), docs/finetune/EVAL.md,
tools/ft_eval.py (SYSTEM prompt, how a heldout-v2 task prompt is built from its spec/skeleton, how candidates are
compiled and tested, code-fence extraction), tools/ft_build_dataset.py (`make_messages`, `validate_solution`,
output row format), tools/ft_leak_check.py.
DO NOT open anything under evals/heldout-v2/ or the legacy held-out task sources: those are the frozen test set and
must stay unseen. You may read tools/ft_eval_heldout_v2.json only for the list of task ids and public names to
AVOID (no reuse of any id, function name, struct name or near-identical problem).
</task>

<tasks>
Layout: data/tasks-v1/<task_id>/ with
- spec.md: the exact user-facing brief (what the model sees), precise: signatures, types, error behaviour,
  edge cases, complexity/allocation rules where relevant.
- skeleton.<c|py> (optional, for skeleton-fill tasks): file with TODO bodies the model must fill.
- solution.<c|py>: reference. Must pass the tests.
- mutants/<n>.<c|py>: >= 1 (aim 2) realistic wrong answers of the kinds in DIAGNOSIS.md (off-by-one, extra
  feature, changed signature, missing error path, wrong rounding, aliasing). Each must FAIL the tests.
- tests/: C tests are a C99 file compiled with `cc -std=c99 -Wall -Wextra -Werror -O1` together with the candidate;
  Python tests are pytest files importing the candidate module (mirror exactly how ft_eval places and names the
  candidate file, including the module filename it expects).
- meta.json: {"id", "language": "c"|"python", "category", "kind": "implement"|"skeleton"|"review",
  "review_notes"?: [..], "previous_attempt"?: path, "source": "codex-batch1"}.
Mix for this batch (50): 18 C99 kernels/utilities with fixtures (bit packing, fixed-point, ring/queue structs,
string parsing, small numeric kernels, quantisation helpers), 18 Python/NumPy utilities with unit tests, 8
skeleton-fill (C or Python), 6 review tasks (spec + previous attempt + precise review notes; reference applies every
note and nothing else). Keep each spec + reference short: spec + solution together under ~6,000 characters
(training rows must stay under ~2,500 tokens to fit memory). Standard library + numpy only.
</tasks>

<tools_to_write>
1. tools/ft_tasks_check.py DIR [--only ID]: for every task, reference passes, every mutant fails, ids unique,
   meta valid, size limit respected; prints a table and exits nonzero on any violation. Reuse ft_eval's test
   runner functions where possible instead of re-implementing them.
2. tools/ft_tasks_to_sft.py DIR --out data/finetune/v1 [--valid-frac 0.1 --seed 20261105]: converts tasks to
   rows in exactly the format of data/finetune/train.jsonl (system = ft_eval.SYSTEM, user = the prompt ft_eval
   would build for an equivalent heldout-v2 task, assistant = one fenced code block with the reference), split by
   task id, writes train.jsonl/valid.jsonl/manifest.json (task ids, sha256 per file), then runs
   tools/ft_leak_check.py and fails if it fails.
3. tools/test_ft_tasks.py: pytest for both tools on 2 tiny fixture tasks under tools/fixtures/ft_tasks/
   (one passing, one with a mutant that wrongly passes => check must fail).
</tools_to_write>

<constraints>
- Python stdlib (+ numpy for tasks). Named constants, functions < 50 lines, no silent excepts.
- Do not modify ft_eval.py, ft_build_dataset.py, ft_leak_check.py, evals/, data/finetune/*.jsonl.
- Do not commit. Gates (run all, paste output): `python3 -m pytest -q tools/test_ft_tasks.py`,
  `python3 tools/ft_tasks_check.py data/tasks-v1`, `python3 tools/ft_tasks_to_sft.py data/tasks-v1 --out
  data/finetune/v1`, `python3 tools/ft_leak_check.py`.
</constraints>

<report>
Files touched; gate outputs; per-category counts; the 5 tasks you consider hardest; anything about ft_eval's prompt
construction you had to guess.
</report>
