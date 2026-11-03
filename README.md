# kimi-k3-c-engine

Use the pure-C CPU engine from [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (C99, no BLAS/framework/GPU, Apache-2.0) to run **~70B-class 8-bit open models** on an ordinary machine, starting with this Mac (M1 Max, 32 GB RAM).

## Why it's a port, not a config change

The engine is hard-wired to Kimi K3. It refuses any other `config.json` on purpose. But Qwen 3.5, 3.6 and 3.8 use the **same kind of hybrid design** as K3, which makes a port realistic:

| | Kimi K3 (engine today) | Qwen3.5/3.6 MoE (target) |
|---|---|---|
| Linear-attention layers | KDA (Kimi Delta Attention) | Gated DeltaNet (same delta-rule family) |
| Full-attention layers | Gated MLA, every ~4th layer | Gated GQA + RoPE, every 4th layer |
| FFN | MoE, sigmoid router, 896 experts top-16 | MoE, softmax router, 256 experts top-8 + gated shared expert |
| Weights | MXFP4 experts | official FP8 e4m3, 128×128 block scales |
| Tokenizer | tiktoken | BPE (`tokenizer.json`) |

Reusable as-is: safetensors reader, disk streaming/mmap, expert cache, sampler, CLI, and the "match a Python reference bit-for-bit" test method.

## Target models (there's no Qwen 3.8 70B)

| Role | Model | 8-bit size | Active / token | Why |
|---|---|---|---|---|
| Dev target | `Qwen/Qwen3.6-35B-A3B-FP8` | 37 GB | ~3B | Smallest of the family, fast iteration, same `qwen3_5_moe` code path |
| **Main target (70B-class)** | `Qwen/Qwen3.5-122B-A10B-FP8` | 127 GB | ~10B | 4× RAM, which is exactly the case the engine's disk streaming exists for |
| Optional dense | `Qwen/Qwen3.8-27B-FP8` | 31 GB | 27B | Same attention layers, dense FFN. Nearly free once the MoE port works |

Baseline: [llama.cpp](https://github.com/ggml-org/llama.cpp) already supports these (`qwen35`, `qwen35moe`) in C/C++. Our engine has to be compared against it at Q8_0 on CPU.

## Layout

```
docs/plan.md                 phased plan + go/no-go
scripts/setup.sh             upstream engine at pinned SHA: build + weightless tests
scripts/setup-baseline.sh    llama.cpp at pinned tag, CPU-only build
scripts/sync.sh              two-way sync of main with GitHub
scripts/install-autosync.sh  launchd job running sync.sh every 5 min (macOS)
engine/                      our port (created in Phase 1)
results/                     one folder per run (metrics committed, raw logs ignored)
third_party/                 upstream engine + llama.cpp (gitignored)
```

## Sync

`main` on GitHub is the single source of truth. Cloud sessions pull at start and push to `main` when done. The local Mac runs `scripts/sync.sh` every 5 min (install with `bash scripts/install-autosync.sh`, remove with `--uninstall`, log at `~/Library/Logs/kimi-k3-c-engine-sync.log`).

## Quick start

```bash
bash scripts/setup.sh            # macOS: brew install libomp first
bash scripts/setup-baseline.sh   # needs cmake
```
