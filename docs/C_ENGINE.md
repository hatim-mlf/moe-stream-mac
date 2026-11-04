# The C engine

A CPU-only inference engine in portable C99 (+ OpenMP, + an ARM NEON kernel) for **qwen3_5_moe** checkpoints
quantized by MLX (affine 4- or 8-bit, group size 32/64/128). It keeps the trunk in RAM and reads the routed
experts from disk on demand, so a 131 GB model runs in ~6 GB of RAM. Math is float32 throughout, which makes it
more exact than MLX's default bf16.

## Build and test

```bash
brew install libomp
make -C engine test          # unit tests against fixtures (no model needed)
make -C engine lib           # engine/build/libqengine.dylib, used by engine/cli/qchat.py
make -C engine build/run_model
```

With a model on disk (`MODEL=~/models/Qwen3.6-35B-A3B-4bit` by default):

| Target | What it checks |
|---|---|
| `make test-real` | 4/8-bit matvec on 6 real tensors (dense and single experts) vs `mx.quantized_matmul` |
| `make test-layer` | one full Gated DeltaNet layer vs mlx-lm, 6 tokens |
| `make test-attn` | one full-attention layer vs mlx-lm, 7 tokens (incl. "cache full" refusal) |
| `make test-moe` | one sparse MoE block vs mlx-lm (256 experts, top-8, shared expert) |
| `make test-model` | whole model: generation + logits at every position vs MLX float32 |

`engine/tests/model_check.py` loads the MLX reference **one layer at a time** in float32, so the whole-model
check fits in RAM even for the 122B.

## Run

```bash
OMP_NUM_THREADS=8 engine/build/run_model <model dir> <prompt.i32> <n_gen> <out.bin>
.venv/bin/python engine/cli/qchat.py --model <model dir> --engine cpu --threads 8
```

`run_model` prints prompt and generation s/token, expert GB read, time spent reading, and peak RSS.
Set `QE_TRACE=<file>` to log every layer's chosen experts (used for the cache study in docs/REPORT.md).

## Code map

| File | Role | Written by |
|---|---|---|
| `src/qmat.c` | matrix × vector on MLX affine 4/8-bit weights; rows split across OpenMP threads; group sums of x computed once | Qwen (3 attempts) + lead |
| `src/qgroup.c` | NEON inner kernel (dot product of one quantized group), scalar fallback | Qwen (from a lead skeleton) |
| `src/qmat_row.c` | dequantize one row (embedding lookup) | Qwen |
| `src/ops.c` | RMSNorm, SiLU·mul, sigmoid, softmax, top-k, RoPE | Qwen (review caught 3 top-k overruns) |
| `src/gdn.c`, `src/gdn_layer.c` | Gated DeltaNet: causal conv, l2norm, gates, delta-rule recurrence; full layer | Qwen |
| `src/attn_layer.c` | gated GQA + partial RoPE + KV cache | Qwen (in a lead skeleton) |
| `src/moe_layer.c` | router softmax → top-k → renormalise → expert SwiGLU MLPs + gated shared expert; parallel expert fetch | Qwen + lead |
| `src/expert_store.c` | `pread` one expert's contiguous byte range from the safetensors file | lead |
| `src/loader.c`, `src/config.c`, `src/model.c` | strict config reader (refuses anything it doesn't fully understand), loader, forward pass | lead + Qwen |
| `src/qengine.c` | C API (`include/qengine.h`) for front ends | lead |
| `src/io/k3_st.c`, `third_party/json.h` | safetensors reader and JSON parser from kimi-k3-in-c (Apache-2.0), with a U32 dtype and rank ≤ 8 | upstream |

## Design notes

- **Streaming.** An MLX expert tensor is `[E][rows][...]`, so expert *e* is one contiguous byte range in each of
  `.weight`, `.scales` and `.biases`. After the router runs, the engine `pread`s exactly those ranges for the 8
  chosen experts, in parallel (OpenMP over 3 × top-k reads, each into its own buffer).
- **Quantization.** `W = q * scale + bias` per group, `q` unsigned. The kernel computes
  `scale * Σ q·x + bias * Σ x` per group, and `Σ x` is shared by all rows.
- **Threads.** 8 is fastest on an M1 Max (10 is slower: the efficiency cores hold the others back). The chat
  defaults to 6 so the Mac stays responsive.
- **Memory.** RSS ~1.4 GB (35B 4-bit) to ~6 GB (122B 8-bit). The OS file cache serves repeat experts for free.

## Speed history (122B 8-bit, s/token)

| Step | s/token |
|---|---|
| 1 thread, scalar | 8.12 |
| 4 threads (OpenMP rows) | 3.69 |
| 8 threads | 3.14 |
| + NEON kernel, group sums once | 1.21 |
| + parallel expert reads | 0.67 |

## Limits and ideas

- Only `qwen3_5_moe` (Qwen 3.5 / 3.6 MoE) with MLX affine quantization. Other families need their own layers;
  the config reader refuses them on purpose.
- No batched prefill: prompts are processed token by token (~1.2 tok/s on the 122B). The MLX hybrid engine
  does batched prefill.
- The ~0.6 s/token floor on the 122B is SSD bandwidth: ~3.8 GB of experts per token.
