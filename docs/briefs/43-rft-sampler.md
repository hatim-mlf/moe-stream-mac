# Brief 43: rejection-sampling fine-tuning (RFT) data from the student's own passing solutions (Codex)

<context>
Supervised fine-tuning of our local Qwen3.6-35B-A3B on 250 test-verified tasks (data/tasks-v1..v5; reference
solutions written by another model) does not improve the frozen heldout-v2 eval: 1-layer LoRA 8/30 vs base
10/30 (paired CI −0.17..0.0), 28 of 30 task outcomes unchanged. The base already predicts the references well
(validation loss 0.228 before training). Next method: sample the *student* on each training task, keep only
its own solutions that pass the task's tests, and fine-tune on those (on-policy data). As a by-product we learn
the base pass rate per training task.
Read: tools/ft_eval.py (`request_model` :89, `extract_code` :84, SYSTEM prompt, `chat_template_kwargs`
enable_thinking False), tools/ft_tasks_check.py (`task_prompt`, `runner_task`, `run_candidate`,
`python_with_test_dependencies`), tools/ft_tasks_to_sft.py (row format, `split_ids`, manifests), and
data/finetune/v5/*manifest.json (the existing train/valid task split).
</context>

<task>
Write `tools/ft_rft.py` with two subcommands (stdlib only; reuse the functions above, do not re-implement them):

1. `sample DIR... --k 8 --temperature 0.7 --seed 20261105 --out results/<dir>/rft-samples.jsonl
   [--endpoint ...] [--limit N] [--only ID]`: for every task folder, build the prompt exactly as
   ft_tasks_to_sft does, request k samples from the running MLX server (base model, no adapter; seed = seed +
   sample index), extract the code, run the task's own tests with the same runner as ft_tasks_check, append
   one JSON line per sample: task_id, batch, sample index, passed, failure tail (≤ 400 chars), reply length,
   seconds. Resumable: skip (task_id, sample) pairs already in the output. A request or test error is a failed
   sample with the error recorded, never a crash of the whole run (but stop with a clear message if the server
   is unreachable for 3 consecutive requests).
2. `build --samples rft-samples.jsonl --tasks DIR... --out data/finetune/rft-v1 [--per-task 2]`: SFT rows in
   exactly the ft_tasks_to_sft format, assistant = a passing student sample (up to --per-task distinct passing
   samples per task, preferring the shortest; exact-duplicate code dropped). Keep the v5 task split: tasks
   listed as valid in data/finetune/v5 manifests go to valid.jsonl, the rest to train.jsonl. Write a manifest
   (sha256 per file, counts) and a `summary.json` with per-task pass@1 estimate (passes/k) and pass@k, the
   distribution (how many tasks at 0, 1..k-1, k passes), and the list of 0-pass tasks. Run ft_leak_check
   and fail if it fails.
</task>

<tests>
`tools/test_ft_rft.py` (pytest, no model): a fake HTTP endpoint (http.server on localhost in a thread) returning
canned replies; two fixture tasks (reuse tools/fixtures/ft_tasks/ if present, else add two tiny ones): one
reply passes, one fails → JSONL lines correct; resume skips done pairs; build keeps only passing samples,
dedups, respects --per-task and the split, refuses when the leak check fails (inject).
Run: `python3 -m pytest -q tools/test_ft_rft.py`.
</tests>

<constraints>
- Do not modify ft_eval.py, ft_tasks_check.py, ft_tasks_to_sft.py, evals/, data/tasks-v*/. Named constants,
  type hints, functions < 50 lines, no silent excepts. No commit.
- Report: files, test output, the exact prompt path you reused, anything unverified.
</constraints>
