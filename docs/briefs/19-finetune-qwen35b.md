<goal>
Prepare a rigorous, evidence-driven fine-tune of our local coding helper, Qwen3.6-35B-A3B (MLX 4-bit, at
~/models/Qwen3.6-35B-A3B-4bit), so that it passes our delegated coding tasks on the FIRST attempt more often and
follows review notes reliably. THIS PHASE: diagnosis, frozen evaluation, dataset pipeline and a feasibility-checked
training recipe. Do NOT start a full training run (it needs the user's OK: the Mac's RAM is shared with Xcode etc.).
Do not commit; the orchestrator commits.
</goal>

<raw_material>
Start from the raw records, not from summaries:
- results/delegations/*/ : every attempt we sent to the model: prompt.md (brief, plus feedback for retries),
  reply.md (model output), stats.json; later ones also system.md. Attempts of one task share a name prefix
  (e.g. qmat-try1..3, gdn-try1..3).
- docs/briefs/*.md : the specs. docs/delegation-log.md : the lead's notes per task (cross-check against raw records).
- git history (git log -p) : the code that was finally accepted for each task.
- Tests that judge each task: engine/tests/test_*.c (make -C engine test), engine/hybrid/test_*.py,
  tools/test_autofix.py (may still be in progress), plus the real-model checks described in docs/C_ENGINE.md.
- tools/helper_rules.md : the rules prompt we now send with every task (prompt-level fixes).
</raw_material>

<deliverables>
1. docs/finetune/DIAGNOSIS.md - an error taxonomy built from the raw records: every failed attempt classified
   (e.g. spec drift / invented semantics, signedness, buffer bounds & aliasing, unused code & -Werror, ignored or
   partially applied review note, wrong restructuring, API misuse, truncated output, control-token issues) with counts
   and links to the attempt folders. For each class: is it fixable by prompt/rules, by the automatic test-feedback
   loop, or only by training? Be explicit and conservative.
2. docs/finetune/EVAL.md + tools/ft_eval.py - a FROZEN evaluation: split tasks into train-eligible vs held-out
   (held-out must never be used for training data), define metrics (first-attempt pass rate, attempts-to-pass with
   the feedback loop, review-note compliance), how to run the baseline on the current model, and paired statistics
   (bootstrap 95% CI over tasks). Also select a small independent public check (e.g. a subset of EvalPlus
   HumanEval+/MBPP+ or LiveCodeBench problems released after the model's training cutoff) runnable offline with
   MLX, to detect both gains and forgetting. tools/ft_eval.py must run end-to-end against the local MLX server
   (scripts/local/serve-mlx.sh, OpenAI API at http://127.0.0.1:8080/v1) with an --adapter option for later.
3. tools/ft_build_dataset.py + docs/finetune/DATA.md - the dataset pipeline:
   (a) extract natural examples from our logs: (brief -> accepted code) for SFT and (brief, failed attempt,
       accepted code) pairs for preference training (DPO/ORPO), plus (brief + code + review note -> corrected code)
       "review-following" examples;
   (b) a generator for MANY new execution-verified tasks in the same style (C99 kernels with fixture tests,
       Python numpy/stdlib utilities with unit tests, skeleton-fill tasks), where reference solutions come from a
       stronger teacher and are KEPT ONLY IF their tests pass (rejection sampling). Target a few hundred examples;
       document the plan even if you only generate a validated seed set of 20-30 now.
   Output JSONL in the format mlx_lm.lora expects (chat "messages"), with train/valid files, and NEVER any held-out task.
4. docs/finetune/RECIPE.md - a training recipe for THIS Mac: mlx_lm.lora (LoRA/QLoRA on the 4-bit model):
   which layers/modules to adapt (start small: attention + shared expert; justify), rank, LR, epochs, batch,
   max sequence length, gradient checkpointing; then optional DPO. Include a measured FEASIBILITY check: a 5-20
   step dry run on the seed set reporting peak memory and s/step (use `/usr/bin/time -l` and mlx's peak memory),
   and state whether it fits next to ~15 GB of other apps. Stop the dry run there.
5. A one-page decision in docs/finetune/README.md: what we expect fine-tuning to fix that rules + the feedback
   loop don't, how big the gain must be to be worth it (with the CI), and the exact next commands.
</deliverables>

<rules>
- Verify, don't assume: run what you write (dataset builder, evaluator on 2-3 tasks, dry run) and paste real outputs.
- Every claim about the model's failures cites the attempt folder it comes from.
- Keep scripts small (functions < ~40 lines), Python 3.14 + mlx/mlx-lm/numpy already in .venv; if you need
  mlx_lm.lora options, read its --help and the installed source instead of guessing.
- Memory: never run more than one model process at a time; the user keeps other apps open.
- Do not modify existing engine code. Do not download anything over 1 GB without saying so in the report first.
</rules>

<report>
Reply with: the taxonomy counts, the train/held-out split, the dry-run memory and s/step, your go/no-go
recommendation for a full run, and the files created.
</report>
