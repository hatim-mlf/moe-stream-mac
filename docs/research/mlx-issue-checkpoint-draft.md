# DRAFT (not filed): mx.checkpoint does not bound backward memory of a chunked recurrence when per-step inputs need gradients

To file on github.com/ml-explore/mlx only after the user approves.

**MLX version:** 0.32.4, macOS 26.4.1, Apple M1 Max 32 GB.

**Summary.** A recurrence is split into chunks of 32 steps, and each chunk is wrapped in `mx.checkpoint`. If the
gradient is taken only with respect to the initial state, backward memory stays flat as the sequence grows, which
is what checkpointing should do. If the gradient is also taken with respect to the per-step inputs (`q, k, v`,
which every training use needs), backward memory grows linearly with the number of steps again: about one
state-sized buffer per step stays alive.

**Repro** (`ckpt_toy2.py`, ~30 lines; state 32×128×128 fp32 = 2 MiB). `python ckpt_toy2.py T plain|ckpt`
prints the backward peak above the inputs:

| T | plain | ckpt (chunk 32) |
|---|---|---|
| 256 | 1783 MiB | 1031 MiB |
| 512 | 3436 MiB | 1582 MiB |
| 1024 | 6510 MiB | 2686 MiB |

The same script with the gradient w.r.t. the initial state only (`ckpt_toy.py`): ckpt 373 / 378 / 427 MiB for
T = 256 / 512 / 1024 (flat), plain 763 / 1296 / 2347 MiB.

**Expected.** With checkpointing, backward memory is O(T/C + C) states, independent of whether the per-step
inputs need cotangents.

**Context.** This blocks memory-efficient LoRA training of Gated DeltaNet models (Qwen3.5/3.6 MoE): the
training path of `mlx_lm.models.gated_delta` (`gated_delta_ops`) keeps ~4 states per token (~8 MiB/token), and
chunked checkpointing only reaches ~6-7 MiB/token.

Attach `results/20261105T143800Z-memory-model/repro/ckpt_toy.py` and `ckpt_toy2.py`.
