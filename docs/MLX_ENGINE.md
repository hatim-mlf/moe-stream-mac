# The MLX hybrid engine

Same idea as the C engine (trunk in memory, experts streamed from SSD), but the math runs on the GPU through
[MLX](https://github.com/ml-explore/mlx)'s Metal kernels. It reuses mlx-lm's model definitions and only replaces
each layer's routed-expert block, so its output is **bit-identical to plain MLX** (bf16).

## Use

```bash
.venv/bin/python engine/cli/qchat.py --model <model dir>                 # --engine gpu is the default
```

From Python:

```python
import sys; sys.path.insert(0, "engine/hybrid")
from hybrid import HybridModel
hm = HybridModel(Path("~/models/Qwen3.5-122B-A10B-8bit").expanduser())
logits = hm.forward(token_ids)        # a whole prompt at once, or one token
hm.reset()                            # new conversation
```

## How it works

1. `mlx_lm.load(..., lazy=True)` builds the model **without reading any weights**.
2. Each layer's `switch_mlp` (the routed experts) is replaced by `StreamingSwitchGLU` before evaluation, so the
   expert tensors are never loaded. Then the trunk is evaluated (loaded) onto the GPU.
3. On each forward pass, the streaming block waits for the router, reads the chosen experts from disk
   (`expert_reader.py`, parallel `preadv`, consecutive experts merged into large disk-ordered reads), and runs
   MLX's `gather_qmm` / `quantized_matmul` on them.
4. **Long batches (≥ 32 tokens, i.e. prompts):** nearly every expert is needed anyway, so each layer's whole
   expert set is read in disk order **once per prompt**, and layer L+1 is read in the background while the GPU
   computes layer L (`LayerPrefetcher`).
5. **Single tokens (replies):** each chosen expert is run straight from its own arrays (no weight copies).

## Files

| File | Role |
|---|---|
| `hybrid.py` | `HybridModel`, `StreamingSwitchGLU`, `LayerPrefetcher` |
| `expert_reader.py` | safetensors index/header parsing, `read` / `read_sorted` (parallel, disk-ordered) |
| `predictive.py` | next-layer expert prediction prefetcher (off by default, see below) |
| `expert_cache.py` | LRU cache of experts in GPU memory (0 GB by default, see below) |
| `hybrid_check.py` | verify against the C engine's output (which matches MLX float32) and measure |
| `hybrid_vs_mlx.py` | verify against plain MLX on a long sequence (35B) |
| `bench_decode.py` | reply speed with cache / prediction options |
| `predict_study.py` | measure next-layer prediction recall |
| `test_*.py` | unit tests for the reader, cache and prefetcher |

## Measured (M1 Max 32 GB, real workload open)

| | 35B 4-bit | 122B 8-bit |
|---|---|---|
| Long prompt (240 tokens) | 10–18 tok/s | 10.4 tok/s (C engine: 1.2) |
| Reply | 6.3–7.4 tok/s | 1.24 tok/s (C engine: 1.5) |
| GPU memory peak | ~2 GB | ~7–10 GB |

## Lessons (each cost a few hours)

- **Cap MLX's buffer cache** (`mx.set_cache_limit(1 GB)`). By default MLX keeps freed buffers for reuse, up to
  most of RAM; with streamed experts that hoarding pushed the Mac into swap and made replies 4–5× slower.
- **A GPU expert cache doesn't pay on a busy 32 GB Mac.** GPU memory and the macOS file cache are the same RAM;
  6–10 GB of cached experts made replies 0.81 → 5.5 s/token. Default 0.
- **Next-layer prediction works but needs reusable buffers.** Running layer L+1's router on layer L's input
  predicts 83% (8 experts), 94% (12) or 97% (16) of layer L+1's choices. As implemented, every layer allocates
  fresh read buffers (~10 GB/token of churn) and macOS compresses the GPU trunk: 3 s/token. Off by default
  (`HybridModel(predict=True)` to try). Fix: preallocated buffers reused per layer, as in the C engine.
- **MLX arrays must be created on the main thread.** Background threads only read bytes into numpy buffers.
- **Python of Anaconda breaks MLX** (it loads Anaconda's MPI and aborts). Use Homebrew Python.

## Ideas

- Reusable read buffers (then re-enable prediction): expected ~0.6 s/token on the 122B.
- Reads that bypass the macOS file cache (`F_NOCACHE`) on machines with no RAM to spare.
- A C++ version on MLX's C++ API, to remove Python overhead and the per-layer sync cost.
