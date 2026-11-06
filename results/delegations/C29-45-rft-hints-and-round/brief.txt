# Brief 45: hinted resampling for unsolved tasks + an automated RFT round (Codex)

<context>
`tools/ft_rft.py` (read it) samples our local Qwen k=8 times per training task (data/tasks-v1..v5) and builds SFT
rows from its own passing solutions. Overnight run (results/20261105T213000Z-rft-base-k8/rft-samples.jsonl, one
JSON line per sample: task_id, batch, sample, k, passed, failure_tail, code, ...): 66% of samples pass; per task
39% are 8/8, 46% are 1-7/8, 16% are 0/8. Fine-tuning on another model's reference solutions made Qwen worse on the
sampled heldout eval (0.320 → 0.147), so training rows must be Qwen's own passing code. A helper
`tools/rft_filter.py: filter_by_pass_rate(samples: dict[str, list[dict]], max_pass_fraction: float) -> dict`
(keeps tasks with pass fraction <= max; being written now, tests in tools/test_rft_filter.py) will exist.
</context>

<task>
1. In `tools/ft_rft.py`:
   - `build --max-pass-fraction F` (default 1.0): drop tasks above F using `rft_filter.filter_by_pass_rate`
     before choosing samples (0.875 drops 8/8 tasks).
   - New subcommand `hint-sample --samples rft-samples.jsonl DIR... --k 4 --out hinted.jsonl`: for every task
     with 0 passes in the samples file, build the normal prompt plus a short appended user note: "A previous
     attempt failed the tests with:\n<failure_tail of that task's first failing sample, ≤ 400 chars>\nWrite a
     corrected solution." Sample, test and record exactly like `sample` (same JSONL fields plus "hinted": true,
     resumable). Important: rows later built from hinted passes must use the ORIGINAL prompt (no hint) as the
     user message — `build` must accept several `--samples` files and treat hinted passes like normal passes.
2. `scripts/rft_round.sh SAMPLES... --name NAME` (bash 3.2 compatible: macOS /bin/bash; no empty-array
   expansion under `set -u`; `set -euo pipefail`; a trap that kills any server it started):
   a. `ft_rft.py build` with `--max-pass-fraction 0.875 --per-task 2` → data/finetune/rft-NAME.
   b. Train two 1-layer LoRA adapters with `tools/ft_lora_lowmem.py --train --config docs/finetune/lora.yaml
      --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --data <dir> --num-layers 1` and env
      `FROZEN_GDN_KERNEL=1 MLX_CACHE_LIMIT_GB=0`: scale 20 and scale 4 (pass a temp yaml copy with
      lora_parameters.scale changed, or the CLI if mlx_lm supports it — check mlx_lm/lora.py), 1 epoch
      (iters = number of train rows), `--val-batches 10 --steps-per-eval <iters/4> --save-every <iters/4>`, into
      results/<UTC>-rft-NAME/scale{20,4}/. Pick per adapter the saved checkpoint with the lowest val loss (parse
      the log; copy it into an adapter dir with adapter_config.json).
   c. For each picked adapter: start `.venv/bin/mlx_lm.server --model ... --adapter-path <dir> --host 127.0.0.1
      --port 8080`, wait for /v1/models, run `tools/ft_eval.py --split heldout-v2 --attempts 1 --temperature 0.6
      --seed 7 --samples 5 --out ...`, stop the server, then `ft_eval.py --paired
      results/20261105T201500Z-eval-resolution/base-s5-t06.json <out>`.
   d. Write results/<UTC>-rft-NAME/metrics.json: dataset counts, per-adapter train time, chosen step, val
      losses, the paired line (base, adapter, delta, CI).
   Refuse to start if anything is listening on port 8080 already.
</task>

<tests>
`tools/test_ft_rft.py` extended (fake endpoint / stubs as now): --max-pass-fraction drops 8/8 tasks;
hint-sample only touches 0-pass tasks, appends the note, records hinted=true, and build uses the original
prompt for hinted passes; multiple --samples files merge. For the script: `bash -n scripts/rft_round.sh` and a
`--dry-run` flag that prints every command without running it (test it).
</tests>

<constraints>
Do not modify ft_eval.py, ft_lora_lowmem.py, mem_guard.py, data/tasks-v*/, evals/. Named constants, type
hints, functions < 50 lines, no silent excepts. No commit. Report: files, test output, the dry-run output.
</constraints>
