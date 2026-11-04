# Fine-tune decision and recommended pipeline

## Decision now: **no-go for full training until two gates pass**

The fine-tune objective is higher first-attempt pass rate on precise test-gated tasks and better review-note application. The historical pre-rules rate was 2/16; after the helper rules and test-feedback loop, one qmat replay passed in one attempt. The frozen current-model baseline has not run, so the present gain available from training is unknown.

The 4-bit model directory is **19 GB on disk**. With 32 GB unified memory, the stated 15 GB of other applications plus a 4 GB reserve already exceeds physical memory before activations and optimizer state. The existing five-step MLX probe did not reach model initialization in this sandbox because there was no Metal device. First close/reduce other applications, measure the actual loaded model, and proceed only if the measured memory gate passes. Do not interpret 3B active parameters as 3B resident weights; all routed experts may be selected.

This research turn made no training run, no dataset download, no code change, and no commit.

## Recommended sequence

### 1. Keep rules, fixture tests, and autofix as the baseline

The existing rules and test loop directly target unused code/`-Werror`, bounds, API shapes, spec drift, and partially applied review notes. Fine-tuning should be measured after those are on. `ft_eval.py` uses the frozen system prompt (including `tools/helper_rules.md`) and runs the frozen fixture gates. The previous qmat rules+autofix replay is evidence that this layer can solve at least one case, not a broad baseline.

On an interactive Metal-enabled Mac, start one server:

```bash
MODEL=/Users/hatim/models/Qwen3.6-35B-A3B-4bit scripts/local/serve-mlx.sh
```

In another terminal, capture the current one-shot, retry, review, and public-mini results:

```bash
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout --attempts 1 --out results/finetune/base-first.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout --attempts 3 --out results/finetune/base-feedback.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split heldout --review-notes --out results/finetune/base-review.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --split public --attempts 1 --out results/finetune/base-public.json
```

**Expected gain:** none from training; this measures the current system. **Go/no-go:** if fewer than two held-out failures are plausibly trainable semantic/review errors, stop: the frozen target requires at least +25 percentage points (two additional passes) to justify training. If the remaining errors are compiler/autofix issues, improve the loop instead. Keep the current no-go if the model will not load or run without sustained memory pressure.

### 2. Build a small, verified, task-disjoint SFT mix

Default to **zero external Hugging Face rows** in the first run. Use the dataset screen in [`DATASETS.md`](DATASETS.md): most external corpora are either unverified for every row, benchmark-like, poorly matched to C99/numerical work, or unclear on source-level licensing. No corpus larger than 1 GB is needed.

First build the already checked-in seeds and accepted train logs:

```bash
.venv/bin/python tools/ft_build_dataset.py
```

For the proposed expanded pilot, create 250 new task candidates from a trusted stronger teacher. The builder executes each candidate's supplied C/Python tests and holds out every fifth accepted task. The tests are co-produced by the teacher, so manually/independently inspect or add assertions before counting a row as execution-verified; reject any underspecified task, weak test, duplicate, held-out keyword/symbol, or candidate that uses non-approved libraries.

```bash
FT_TEACHER_API_KEY=<key-if-required> .venv/bin/python tools/ft_build_dataset.py \
  --teacher-url https://<trusted-teacher>/v1/chat/completions \
  --teacher-model <stronger-coding-model> \
  --teacher-count 250 --max-rejections 5
```

The expected builder split is 200 teacher train and 50 teacher validation rows. With the existing data this produces **231 train rows** (200 teacher + 22 validated seeds + 6 accepted train logs + 3 review-follow corrections) and **60 validation rows** (50 teacher + 6 seeds + 2 natural validation + 2 review validation). The existing manifest builder asserts that held-out task IDs never enter train, validation, preference, or review files. Inspect `data/finetune/manifest.json` after generation.

**Expected gain:** more correct worked examples for C bounds/signedness, Python/NumPy shapes, exact APIs, and review corrections. **Go/no-go:** train only if all selected rows have a specific prompt, deterministic independent tests, correct accepted output, no exact function-name overlap with `tools/ft_eval_manifest.json` targets, and task-level train/validation separation. If an independent test audit is not available, use only the checked-in verified seeds/logs and keep the claim narrower.

### 3. Measure the five-step memory/throughput probe

Stop the inference server before probing. Current measured MLX memory is unknown. The minimum physical-memory condition is approximately `measured MLX peak + other-app use + 4 GB reserve < 32 GB`; the 19 GB on-disk model suggests other apps must be well below the current ~15 GB, and the real loaded footprint may be larger. Do not proceed if Metal initialization fails, memory pressure becomes sustained, or the probe cannot run beside the intended applications.

```bash
/usr/bin/time -l .venv/bin/python tools/ft_lora_probe.py \
  --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit \
  --config docs/finetune/lora.yaml \
  --data data/finetune/probe \
  --train --iters 5 --batch-size 1 --max-seq-length 10240 \
  --grad-checkpoint --steps-per-report 1 \
  --adapter-path /tmp/qwen35-ft-probe
```

**Expected gain:** none; this measures peak memory and seconds/step. **Go/no-go:** require all five steps, recorded peak plus headroom under 32 GB, no persistent swap/memory pressure, and a feasible projected duration. The previous attempt completed 0/5 because MLX had no Metal device, so it provides no fit estimate.

### 4. Run masked QLoRA-style SFT only if steps 1–3 pass

The current recipe adapts the last four blocks: three Gated DeltaNet blocks plus one full-attention block, along with their shared experts. It freezes the 256 routed experts and router. Rank 8 keeps the recommended adapter at about 1.12M parameters; the base is quantized affine 4-bit, so this is QLoRA-style adaptation. The checked-in config uses prompt masking, gradient checkpointing, batch 1, `1e-5` learning rate, and MLX `scale: 20` (direct multiplier; equivalent to conventional α=160 at rank 8). This scale has not been validated against a smaller `scale: 2` ablation.

For 231 training rows, three epochs are **693 iterations**. Stop the server, then run:

```bash
.venv/bin/mlx_lm.lora \
  --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit \
  --config docs/finetune/lora.yaml \
  --data data/finetune --train --iters 693 \
  --batch-size 1 --max-seq-length 10240 \
  --grad-checkpoint --mask-prompt --val-batches -1 \
  --steps-per-eval 231 \
  --adapter-path adapters/qwen35-pilot
```

**Expected gain:** improve completion-token behavior on semantics and exact review correction; no guaranteed gain on compiler-only errors or spec omissions absent from the data. **Go/no-go:** this is the actual run only after the five-step peak-memory gate. Stop early if validation loss rises sharply or the first epoch introduces memory pressure. Three epochs is a ceiling, not a target to extend automatically.

After training, restart the server with the adapter and run the exact frozen evaluation:

```bash
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split heldout --attempts 1 --out results/finetune/adapter-first.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split heldout --attempts 3 --out results/finetune/adapter-feedback.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split heldout --review-notes --out results/finetune/adapter-review.json
.venv/bin/python tools/ft_eval.py --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit --adapter adapters/qwen35-pilot --split public --attempts 1 --out results/finetune/adapter-public.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-first.json results/finetune/adapter-first.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-feedback.json results/finetune/adapter-feedback.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-review.json results/finetune/adapter-review.json
.venv/bin/python tools/ft_eval.py --paired results/finetune/base-public.json results/finetune/adapter-public.json
```

Keep the adapter only if first-attempt held-out pass rate improves by at least **25 points**, the paired 95% bootstrap lower bound is above zero, pass-at-cap/review-note acceptance do not decrease, and the public mini does not regress. Eight held-out tasks are a pilot, not evidence for a broad coding claim.

### 5. Consider preference training only after a real pair bank exists

`preferences.jsonl` contains three train-only pairs today: the passing accepted answer is chosen and the first failed answer is rejected. Three pairs are insufficient. Collect at least 50 distinct train-only pass/fail pairs, all from tasks outside validation/held-out, and confirm the rejected answer really fails the frozen suite. The chosen/rejected distinction must come from tests, not an LLM judge.

ORPO is the first future preference objective to evaluate because it avoids a reference model; DPO is a reasonable alternative if reference log-probabilities can be cached. But installed `mlx-lm==0.32.2` has no DPO/ORPO/KTO/IPO CLI or trainer, and this environment has no MLX preference package installed. **There is no valid preference-training command in this repo yet.** Do not run PyTorch TRL on these MLX weights or invent a command; add and memory-probe an MLX-compatible implementation in a separate change before considering this step. GRPO is not feasible to recommend on this 32 GB Mac; rejection-sampling task generation is the practical execution-feedback method.

## Files and current evidence

- [`DATASETS.md`](DATASETS.md) screens public datasets, licenses, execution labels, format, benchmark contamination, and the recommended 231/60 train/validation mix.
- [`METHODS.md`](METHODS.md) gives the objectives, module/parameter counts for the local config, MoE/quantization implications, and installed trainer support.
- Existing supporting records: [`DIAGNOSIS.md`](DIAGNOSIS.md), [`EVAL.md`](EVAL.md), [`DATA.md`](DATA.md), [`RECIPE.md`](RECIPE.md), [`lora.yaml`](lora.yaml), [`tools/helper_rules.md`](../../tools/helper_rules.md), and [`tools/ft_eval_manifest.json`](../../tools/ft_eval_manifest.json).
- The frozen public mini is a regression check only; the Qwen model card does not state a training-data cutoff. Do not describe it as contamination-resistant.
