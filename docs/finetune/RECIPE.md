# MLX LoRA recipe and feasibility

## Starting configuration

This uses the local quantized model at `/Users/hatim/models/Qwen3.6-35B-A3B-4bit` and installed MLX-LM 0.32.2
(MLX 0.32.4, Python 3.14). Quantized base weights remain frozen; `mlx_lm.lora` converts selected quantized
linear layers to LoRA layers, so this is QLoRA-style adaptation. The model card describes 40 layers arranged as
ten groups of three Gated DeltaNet blocks followed by one full-attention block, with one shared MoE expert per
block. Adapting the last four blocks includes both attention types and the shared expert while leaving routed
experts, router, embeddings, norms, and output head frozen. [Model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).

The installed `mlx_lm/tuner/utils.py` confirms `lora_parameters.keys` is matched against each layer's local
module names. `docs/finetune/lora.yaml` therefore names the linear-attention projections, full-attention
projections, and shared-expert projections/gate. The start-small settings are rank 8, scale 20, dropout 0,
learning rate `1e-5`, batch 1, gradient checkpointing, prompt masking, and max sequence length 10,240. The local
tokenizer measured a longest current train example of 5,269 tokens and validation example of 9,153, so 4,096
would truncate current examples. Three passes over the 31 current train rows are 93 steps; evaluate every 31
steps with two validation batches. Recalculate the step count and token lengths after teacher generation.

`mlx_lm.lora` supports `lora`, `dora`, and full tuning; its installed source does not implement DPO/ORPO. The
preference sidecar is ready for a later compatible MLX preference trainer, but that trainer needs its own
version check and memory probe before use. The CLI `--help` also aborts during MLX import in this sandbox; option
names and layer-key behavior here were checked in the installed `mlx_lm/lora.py` parser and `mlx_lm/tuner/utils.py`.

## Five-step memory probe

The builder writes `data/finetune/probe/` with four short validated seeds plus the longest train-eligible
review example in `train.jsonl`, and the longest validation review example in `valid.jsonl`. The exact chat
template lengths are 5,269 and 9,153 tokens. `tools/ft_lora_probe.py` wraps the installed `mlx_lm.lora` entry
point, times only the training loop for aggregate seconds/step, resets MLX peak memory before model loading,
and prints `mx.get_peak_memory()` afterward. `/usr/bin/time -l` adds process resident-memory data.

Command attempted in this workspace:

```bash
/usr/bin/time -l .venv/bin/python tools/ft_lora_probe.py \
  --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit \
  --config docs/finetune/lora.yaml \
  --data data/finetune/probe \
  --train --iters 5 --batch-size 1 --max-seq-length 10240 \
  --grad-checkpoint --steps-per-report 1 \
  --adapter-path /tmp/qwen35-ft-probe
```

It did not reach model loading or execute a training step. Real output:

```text
Traceback (most recent call last):
  File "/Users/hatim/moe-stream-mac/tools/ft_lora_probe.py", line 5, in <module>
    import mlx.core as mx
ImportError: [metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
        0.05 real         0.03 user         0.01 sys
time: sysctl kern.clockrate: Operation not permitted
```

| Measurement | Result |
|---|---|
| Requested dry-run steps | 5 |
| Completed training steps | 0; import stopped before MLX device initialization |
| MLX peak memory | Not measured; `mx.get_peak_memory()` was never reached |
| Seconds/step | Not measured |
| `/usr/bin/time -l` peak RSS | Not available; time exited after `sysctl kern.clockrate: Operation not permitted` |
| Fits beside about 15 GB of other apps? | Unknown; no device-memory measurement was possible |

This is a **no-go for a full training run** until this five-step command succeeds on the interactive Metal-enabled
Mac. After it runs, record the emitted `mlx_peak_memory_gib`, `s_per_step`, and `/usr/bin/time -l` maximum RSS.
For this workload, require measured MLX peak plus 15 GB for other apps plus a 4 GB OS/runtime reserve to remain
below physical unified memory, with no sustained memory-pressure warning. If it fails that budget, reduce trainable
layers first, then sequence length only after splitting/trimming examples and repeating the length audit.

## Full pilot command, only after a separate go decision

Stop the inference server before training so no second model remains resident. With today's 31 training rows,
93 iterations is three epochs at batch 1:

```bash
.venv/bin/mlx_lm.lora \
  --model /Users/hatim/models/Qwen3.6-35B-A3B-4bit \
  --config docs/finetune/lora.yaml \
  --data data/finetune --train --iters 93 \
  --adapter-path adapters/qwen35-pilot
```

Do not run this full command as part of the current phase. After training, restart the local server and run the
paired held-out, review-note, and public evaluations in [`EVAL.md`](EVAL.md). Only consider preference tuning if
the SFT adapter improves the frozen metrics without public regression and a compatible trainer is separately
feasibility-checked.
