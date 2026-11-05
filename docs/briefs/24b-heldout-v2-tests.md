<task>
Revision of the heldout-v2 suite in this worktree (evals/heldout-v2/, built from docs/briefs/24-heldout-v2.md; read
that brief first). The orchestrator's audit REJECTED the tests as too thin: 520 test lines for 30 tasks. Example:
evals/heldout-v2/c_u4_groupdot/tests/test_c_u4_groupdot.c checks one valid case and one invalid length, while its
spec also requires: null-pointer rejection for each pointer, group <= 0, n <= 0, n % group != 0, out unchanged on
every error, unsigned nibbles (values >= 8 matter), multiple groups with different scales/biases, larger n.
A weak test lets wrong answers pass and makes the evaluation meaningless. Specs, ids, task.json kinds/difficulty
and references stay as they are unless a spec is ambiguous (then fix the spec minimally and say so).
</task>

<required_for_every_task>
1. Every requirement sentence in spec.md is checked by at least one named test case: every error path / exception
   type, every "unchanged on error" promise, every edge case (empty/zero/limits/ties/aliasing/dtype/bool-vs-int),
   and the exact return type/dtype/shape.
2. A randomized property test against an INDEPENDENT naive implementation written inside the test (not the reference
   file): fixed seed, >= 200 cases over several sizes (including sizes that are not multiples of 4/8/16, and the
   largest size the spec allows that runs in < 2 s). Float comparisons: relative tolerance 1e-5 plus absolute 1e-6
   (state it in the test); integer/bit outputs exact.
3. Failure messages name the case, the index and expected vs got (they are fed back to the model on retries).
4. tests/COVERAGE.md: a table mapping each spec requirement -> test case name.
5. At least 4 mutants per task (keep the 2 existing), each violating a DIFFERENT requirement, all realistic
   DIAGNOSIS.md-style mistakes; every mutant must fail. Add at least one mutant that only breaks an error path and
   one that only breaks the randomized/general case.
6. Skeleton tasks: the test still verifies that non-TODO lines are byte-identical to the template.
7. Runtime per task < 10 s on a laptop CPU.
</required_for_every_task>

<constraints>
No Metal, no localhost model access; you can compile C and run .venv/bin/python. Touch only evals/heldout-v2/,
tools/ft_eval_heldout_v2.json (re-freeze) and docs/finetune/EVAL.md (update counts). Do NOT git add or commit.
</constraints>

<verification_loop>
  .venv/bin/python tools/ft_heldout_v2_freeze.py
  .venv/bin/python tools/ft_eval.py --split heldout-v2 --self-check        # 30/30
  .venv/bin/python tools/ft_eval.py --split heldout-v2 --mutant-check      # >= 120 mutants, all rejected
  .venv/bin/python tools/ft_leak_check.py
  .venv/bin/python tools/test_ft_eval_v2.py
Also report: total test lines, and min/median test lines per task.
</verification_loop>

<structured_output_contract>
1. Per task: number of test cases, number of random cases, number of mutants, any spec change.
2. Gate outputs. 3. Anything you want a decision on.
</structured_output_contract>
