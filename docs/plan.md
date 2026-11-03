# Plan: port the C engine to Qwen 3.5/3.6 MoE (8-bit)

Every result folder records the engine commit, model + revision, machine, prompt, token count and the full command.

## Phase 0: Ground truth (done on Mac, see `results/`)

- Upstream engine builds and passes its weightless tests on M1 Max.
- Build the llama.cpp CPU baseline (`scripts/setup-baseline.sh`).

## Phase 1: Fork and reference (cloud session OK, no big downloads)

- Copy the reusable upstream files into `engine/`, keeping the Apache-2.0 notice and adding a NOTICE that credits upstream. Reusable files: safetensors reader, streaming/mmap loader, expert cache, ops, sampler, CLI.
- Write `tools/qwen_ref.py`, a tiny NumPy reference of the `qwen3_5_moe` forward pass, checked against HF `transformers` on a **randomly initialised tiny config** (2–4 layers, few experts). Then generate fixtures the same way upstream does (`tests/fixtures`).
- Pass: reference logits match `transformers` to within 1e-4.

## Phase 2: Implement the new pieces in C, one gated test each

1. FP8 e4m3 dequant matmul with 128×128 block scales (NEON + scalar fallback).
2. Gated DeltaNet layer: short conv1d, per-head decay (`A_log`, `dt_bias`), delta-rule recurrence, gated RMSNorm output.
3. Gated GQA full attention: q/k norm, RoPE, KV cache.
4. MoE: softmax top-k router, renormalisation, shared expert with sigmoid gate.
5. Config reader for `qwen3_5_moe` that refuses anything it doesn't fully understand (keep upstream's strictness).
6. Tokenizer: start with Python pre-tokenisation (`--ids`), and port BPE to C later.

Pass: tiny-model fixtures match the reference exactly (teacher forcing, greedy, incremental decode), just like upstream's gates.

## Phase 3: Real models on the Mac (approval needed before each download)

Disk has ~252 GB free, so download one model at a time and delete it after measuring.

1. `Qwen3.6-35B-A3B-FP8` (37 GB): first real tokens, then compare with `transformers` on 5 prompts (top-1 agreement), and with llama.cpp Q8_0.
2. `Qwen3.5-122B-A10B-FP8` (127 GB): the 70B-class target. It's bigger than RAM, so disk streaming and the expert cache do the work.

Measure tokens/s (prefill and decode, mean of 3 after warm-up), peak RSS and disk read throughput, for **our engine vs llama.cpp Q8_0 CPU-only** on the same prompts.

## Go/no-go

- **Win**: on the 122B model with 32 GB RAM, our engine is faster or uses less memory than llama.cpp, or runs where llama.cpp can't. If so, keep going (chat mode, dense 27B, upstream PR or a standalone release).
- **Learned-only**: correct but slower than llama.cpp. Write up the findings and stop.
- **Stop early**: Phase 2 kernels don't match the reference after reasonable effort.
