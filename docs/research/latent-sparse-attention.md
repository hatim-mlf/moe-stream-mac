# Latent Sparse Attention (LSA): parked for later

Checked 2026-11-08 by Claude from the repo README (not the paper). Status: **parked**, no work planned.

## What it is

- Repo: <https://github.com/Prefrontal-AI/latent-sparse-attention> (Apache-2.0, created 2026-10-27).
- Paper: "Latent Sparse Attention: Towards Subquadratic Ultra-Long Context Attention".
- Keys are pooled and clustered into buckets. Each query scores the bucket centroids, then attends only to the
  selected keys plus the sink and window. D-LSA distills an indexer onto an existing model; published maps exist
  for Qwen3.8-27B (release `d-lsa-weights-v1`).

## Claims vs evidence

- 72.4x prefill / 42.5x decode on Qwen3.8-27B: **one attention layer** measured alone vs full attention, in a
  128k-10M token benchmark (context for the Qwen figure not stated). Not end to end.
- "9x first token" (from a social post): not in the README; unverified.
- Accuracy: RULER only, Qwen3.8 D-LSA 0.974 (4k) and 0.968 (32k); the README shows no dense baseline number
  next to it. No coding benchmark.
- Hardware: NVIDIA H200/B300, CUDA, the authors' SGLang fork. No MLX/Metal.

## Why it doesn't help us now

Qwen3.8-27B has 16 full-attention layers and 48 linear-attention layers; KV cache is 64 KiB/token
(16 layers x K,V x 4 KV heads x 256 dim x fp16). On the M1 Max (32 GB), decode reads ~16 GB of weights per
token, so attention's share, and LSA's best-case decode speedup, is:

| Context | KV cache | Attention share | Max speedup |
|---|---|---|---|
| 16k | 1 GB | ~6% | ~1.06x |
| 32k | 2 GB | ~11% | ~1.1x |
| 128k | 8 GB | ~34% | ~1.5x |
| ~180k (RAM limit) | ~12 GB | ~43% | ~1.7x |

LSA still stores every key, so it does not raise the RAM context ceiling.

## The idea worth revisiting

Combine LSA with this project's SSD streaming: keep bucket centroids in RAM and the full KV cache on SSD, and read
only the selected buckets each step. At a ~2k-key budget that is ~128 MB/token, ~20-25 ms at M1 Max SSD speed,
which could take context past RAM toward the model's 262k limit. Estimate only, not measured.

Port cost: indexer plus sparse-gather kernel in MLX (`mx.fast.metal_kernel`), mlx_lm Qwen3.8 integration,
conversion of the PyTorch maps, and a RULER accuracy re-check. Weeks of engine work.

Cheaper first step if long context is ever needed: mlx_lm KV quantization (`--kv-bits 8` or `4`), 2-4x context
in the same RAM.
