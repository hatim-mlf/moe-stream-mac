<task>
Build a second frozen held-out coding evaluation, "heldout-v2": 30 NEW tasks, each with a frozen spec, automatic
tests, a reference solution that passes them, and at least 2 plausible wrong solutions ("mutants") that the tests
reject. Wire it into tools/ft_eval.py as a new split. These tasks measure a local model (Qwen3.6-35B-A3B, 4-bit)
before and after weight edits / fine-tuning; they must NEVER be usable as training data.

Read first: docs/finetune/EVAL.md, docs/finetune/DIAGNOSIS.md (the model's failure classes), tools/ft_eval.py,
tools/ft_eval_manifest.json, tools/helper_rules.md (the system prompt rules every task is answered under),
tools/ft_build_dataset.py, and two example frozen specs:
results/delegations/20261104T181100Z-pinned-try1/prompt.md and results/delegations/20261103T205114Z-qmat-try1/prompt.md.
</task>

<layout>
One directory per task: evals/heldout-v2/<id>/ with
  spec.md        the exact prompt the model receives (self-contained; style of the pinned example: target file
                 name and language line, exact signatures, behaviour incl. edge cases and error codes, "Return only
                 the complete file in one fenced <lang> block."). Never contains the solution.
  task.json      {"id", "lang": "c"|"python", "target": "<file the candidate is written to, relative to the task
                 dir>", "build_cmd": [...] or null, "test_cmd": [...], "timeout_s": <=60, "classes": [DIAGNOSIS
                 class names it probes], "kind": "kernel"|"utility"|"skeleton", "difficulty": "easy"|"medium"|"hard"}
                 Commands are argv lists run with cwd = a temp copy of the task dir; "{python}" is replaced by the
                 repo .venv python and "{cc}" by cc.
  tests/         the test program(s). C: a test_<id>.c with main() that returns nonzero on any failure and prints
                 which case failed and why (that output is fed back to the model in retry mode, so make it
                 specific: case name, index, expected vs got). Python: a plain script with asserts or unittest.
  fixtures/      optional, deterministic, < 1 MB per task, produced by a committed gen_fixtures.py (numpy, fixed
                 seed). Prefer computing expectations inside the test from an independent naive implementation or
                 hand-checked vectors when that is simpler.
  reference/     the reference solution (same file name as target).
  mutants/       >= 2 wrong solutions, each a realistic mistake from DIAGNOSIS.md (sign-extended nibble, group index
                 off by one, top-k overrun, aliasing assumption, wrong axis, dtype drift, ignored edge case, unused
                 variable under -Werror, wrong error code, ...). Each must COMPILE (except a deliberate -Werror
                 mutant) and FAIL the tests.
  review.md      ONLY for 10 of the tasks (mix of C and Python): a review-note prompt = the spec, then one mutant
                 presented as "Your previous attempt", then 1-3 precise review notes that, applied exactly, fix it.
                 Tests decide pass/fail.

C build flags: exactly the engine's: cc -O2 -std=c99 -Wall -Wextra -Werror ... -lm. Scalar C99 only (no NEON, no
OpenMP), no malloc in kernels unless the spec says so. Python: 3.14, stdlib + numpy only.
</layout>

<task_mix>
30 tasks total, roughly: 12 C99 kernels (quantized 4/8-bit unpack + dot with per-group scales/biases, int8/uint8
saturation, top-k with ties and bounds, ring buffers, in-place vs out-of-place with documented aliasing rules,
strided/transposed matvec, softmax/rmsnorm/logsumexp numerics, bit packing, prefix sums, small parsers that return
error codes); 12 Python/NumPy utilities (shape/dtype contracts, broadcasting, axis handling, stable sorting with
lexsort ties, segment ops, sliding windows, byte-layout parsing with struct, caches with byte budgets, error
behaviour); 6 skeleton-fill tasks (the spec contains a whole file with TODO bodies and fixed signatures; the model
fills only the TODOs; the test also checks that non-TODO lines are unchanged, e.g. by comparing the file with TODO
regions stripped).
Spread difficulty so a strong-but-imperfect 35B coder would pass about 20-50% on the first attempt: ~1/3 easy,
~1/3 medium, ~1/3 hard. Hard means many precise spec details, not obscure algorithms.
Novelty: invent the tasks; do not copy HumanEval/MBPP/LeetCode/EvalPlus problems or the repository's own functions.
No id or public function name may equal any in tools/ft_eval_manifest.json, data/finetune/, or engine/src.
</task_mix>

<harness_changes>
1. Manifest: tools/ft_eval_heldout_v2.json listing every task with sha256 of spec.md, review.md (if any),
   task.json, and every file under tests/ and fixtures/ (NOT reference/ or mutants/), plus a suite_sha256 over the
   sorted list. Write tools/ft_heldout_v2_freeze.py to (re)generate it; ft_eval refuses to run if any hash differs.
2. tools/ft_eval.py: new --split heldout-v2 (one response per spec.md; with --review-notes, one per review.md).
   Generic runner: copy the task dir (without reference/ and mutants/) to a temp dir, write the candidate at
   target, run build_cmd then test_cmd with the task timeout. Same retry/feedback loop and same result schema as
   the legacy split; record "suite": "heldout-v2" and suite_sha256 in the output. --self-check on this split runs
   each reference (must pass); new --mutant-check runs every mutant (must fail) and prints a table.
   Leave the legacy "heldout" split, FROZEN list, PUBLIC tasks and the frozen SYSTEM prompt byte-identical (the
   system-prompt hash check must still pass), so results/finetune/baseline-heldout-a1.json stays pairable.
3. Sampling: add --temperature (default 0), --seed (optional, sent as "seed" in the request body only if given),
   --samples K (default 1: K independent runs per task). Record all three in the output. With K > 1 each task row
   gets "samples": [...per-sample rows...] and its first_pass / passed become the per-sample mean (floats).
   paired_report: refuse to pair files that differ in suite, suite_sha256/benchmark_revision, temperature, samples
   or task ids; bootstrap over tasks on the per-task values (unchanged method and seed). Also print n tasks and the
   mean of each side, not just the delta.
4. Leak guard: tools/ft_build_dataset.py derives its held-out ids from BOTH manifests (legacy HELDOUT set plus every
   heldout-v2 id) instead of a hardcoded set, and also rejects any row whose text contains a heldout-v2 public
   function name. Add tools/ft_leak_check.py that scans data/finetune/ and evals/ for: heldout-v2 ids or function
   names appearing outside evals/heldout-v2/, and exits nonzero with the offending file:line.
5. Document the new split in docs/finetune/EVAL.md (a short section: purpose, layout, mix table with ids/kind/
   difficulty/classes, commands, the never-train rule).
</harness_changes>

<constraints>
- Your sandbox has no Metal GPU and no localhost network: you cannot call the model. You CAN compile C and run
  .venv/bin/python; do run every gate below yourself.
- You are working in a git worktree on branch heldout-v2. Do not touch engine/, third_party/, results/, the legacy
  manifest, or tools/helper_rules.md. Download nothing.
- Do NOT git add / commit. The orchestrator reviews and commits.
</constraints>

<verification_loop>
Run and fix until all pass:
  .venv/bin/python tools/ft_heldout_v2_freeze.py
  .venv/bin/python tools/ft_eval.py --split heldout-v2 --self-check          # 30/30 references pass
  .venv/bin/python tools/ft_eval.py --split heldout-v2 --mutant-check        # every mutant fails
  .venv/bin/python tools/ft_eval.py --split heldout --self-check             # legacy 8/8 still pass
  .venv/bin/python tools/ft_eval.py --split public --self-check
  .venv/bin/python tools/ft_leak_check.py
  .venv/bin/python tools/ft_build_dataset.py --help >/dev/null
  make -C engine test
Also add a tiny unit test tools/test_ft_eval_v2.py (paired_report refuses mismatched temperature/suite; per-sample
means; manifest hash mismatch is refused) and run it.
</verification_loop>

<structured_output_contract>
End with:
1. Table of the 30 tasks: id, lang, kind, difficulty, classes, number of mutants, has review.md.
2. Files touched.
3. Gate outputs (counts).
4. Anything you deviated on or want a decision on.
</structured_output_contract>
