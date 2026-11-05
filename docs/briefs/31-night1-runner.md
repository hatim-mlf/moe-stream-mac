<task>
Write `tools/ft_night.py`: one unattended overnight pipeline for LoRA fine-tuning of our local Qwen3.6-35B-A3B
(MLX 4-bit at /Users/hatim/models/Qwen3.6-35B-A3B-4bit, mlx-lm 0.32.2, MLX 0.32.4, Python 3.14, venv at .venv).
The user starts it and goes to bed. Nobody is watching: it must never prompt, never use sudo, never hang, and
always leave the machine with no MLX process running and a metrics file explaining what happened.
Read first: docs/finetune/RECIPE.md, docs/finetune/lora.yaml, tools/ft_lora_probe.py, tools/ft_eval.py
(especially `arguments`, `--adapter`, `--paired`), tools/ft_build_dataset.py (do not modify it),
scripts/local/serve-mlx.sh, CLAUDE.md.
</task>

<stages>
Run in order; any failure stops the pipeline, records which stage failed and why, and runs cleanup.
1. preflight
   - `sysctl -n iogpu.wired_limit_mb` must be >= 28672 (constant). If not, exit with:
     "run: sudo sysctl iogpu.wired_limit_mb=28672" in the log and metrics.
   - kill any `mlx_lm.server` / `mlx_lm.lora` process (pkill -f), wait until gone (timeout).
   - free disk >= 15 GB on the results volume; data files exist; record sha256 of train.jsonl, valid.jsonl,
     data/finetune/manifest.json and lora.yaml into metrics.
   - run `tools/ft_leak_check.py` (read its CLI) on the training data; nonzero exit = abort.
2. probe: run `tools/ft_lora_probe.py` exactly as RECIPE.md describes (5 steps, probe data). Record peak memory and
   seconds/step. Abort if it fails or peak memory > 27.5 GB (constant).
3. train: `mlx_lm.lora` with docs/finetune/lora.yaml, overriding `iters` from the CLI flag `--iters`
   (default 93 = 3 epochs of 31 rows) and `adapter_path` = <run_dir>/adapter. Data dir default data/finetune,
   flag `--data`. Stream stdout/stderr to <run_dir>/train.log. Parse train/val loss lines into metrics
   (list of {step, train_loss | val_loss}). Hard wall-clock timeout flag `--train-timeout-min` (default 300).
   The process that loads weights must call `mx.set_wired_limit(...)` (= the recommended max working set) before
   loading, or macOS compresses the weights and it stalls: if `mlx_lm.lora` is run as a subprocess, launch it via
   a tiny wrapper (`python -c` or a helper module) that sets the wired limit then calls `mlx_lm.lora.main()`.
   Same for the server in stage 4.
4. eval: start `mlx_lm.server --model <model> --adapter-path <run_dir>/adapter --host 127.0.0.1 --port 8080`
   (same wired-limit wrapper), poll `/v1/models` until healthy (timeout 600 s). Then run, writing into run_dir:
   - `ft_eval.py --split heldout-v2 --attempts 1 --out <run_dir>/adapter-v2-first.json`
   - `ft_eval.py --split public --attempts 1 --out <run_dir>/adapter-public.json`
   - `ft_eval.py --paired results/finetune/base-v2-first.json <run_dir>/adapter-v2-first.json` (capture text)
   Use whatever flags ft_eval actually needs for the adapter (read `--adapter`; prefer the server-side
   --adapter-path so the field is not needed, but record it in the result file if ft_eval requires it).
   Flag `--skip-eval` skips this stage.
5. cleanup (always, also on failure / SIGINT / SIGTERM): stop the server and any child, verify no mlx process of
   ours remains. Do NOT restart the user's Qwen server.
</stages>

<outputs>
run_dir = results/<UTC yyyymmddTHHMMSSZ>-night1/ (flag `--name` replaces "night1").
- metrics.json: {status: "ok"|"failed", failed_stage, error, started, finished, per-stage seconds, hashes,
  probe {peak_gb, s_per_step}, train {iters, losses, adapter_path, seconds}, eval {v2_first passed/total,
  public passed/total, paired_text}, wired_limit_mb, git_head}. Written after every stage (partial is fine).
- night.log: timestamped stage lines.
- Never write weights outside run_dir/adapter. Add `results/*/adapter/` to .gitignore.
</outputs>

<flags>
--dry-run: run preflight checks except the wired-limit requirement (report it as a warning), validate the data,
lora.yaml and command lines, and print the exact commands of stages 2-4 without running them; with
`--dry-run --eval-only-endpoint URL` additionally run stage 4's ft_eval calls with `--limit 2` against an
already-running server at URL (no server start/stop). This is how the lead tests it in the daytime.
--iters, --data, --name, --model, --train-timeout-min, --skip-eval.
</flags>

<tests>
tools/test_ft_night.py (pytest, no MLX, no network): stage ordering and abort-on-failure with stubbed
subprocess runner; wired-limit check (stub sysctl output below/above threshold); loss-line parsing on real-format
sample lines from mlx_lm.lora (read its source for the exact print format: "Iter N: Train loss X" and
"Iter N: Val loss X"); cleanup runs on exception; metrics.json written after a failed stage; dry-run prints
commands and starts no process. Use dependency injection for subprocess/sysctl so tests never touch the system.
</tests>

<constraints>
- Python 3 stdlib only in ft_night.py (subprocess, signal, json, pathlib, argparse, hashlib, shutil, time).
- Named constants for every threshold. Functions < 50 lines. No silent excepts: log the exception.
- You cannot run MLX or reach localhost in your sandbox; do not try. The lead runs it on the Mac.
- Do not modify ft_eval.py, ft_build_dataset.py, ft_lora_probe.py, lora.yaml or anything in evals/.
- Do not commit. Gate: `.venv/bin/python -m pytest -q tools/test_ft_night.py` (use `python3 -m pytest` if the
  venv python is unavailable to you).
</constraints>

<report>
Files touched; exact gate output; the command lines stage 2-4 will run; anything you assumed about mlx_lm.lora
or ft_eval CLI that you could not verify from source.
</report>
