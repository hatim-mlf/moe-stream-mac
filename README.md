# kimi-k3-c-engine

Run a strong ~30B-class Qwen model locally on this Mac (M1 Max, 32 GB) and let Claude Code hand work to it. There are two tracks:

| Track | What | Status |
|---|---|---|
| **A: Local worker** ([plan](docs/local-agent-plan.md)) | Serve Qwen with MLX or llama.cpp, benchmark it, and plug it into Claude Code through an MCP bridge | Scripts ready, needs a run on the Mac |
| **B: C engine port** ([plan](docs/plan.md)) | Port the [kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) CPU engine to Qwen MoE, so **8-bit models bigger than RAM** stream from SSD | Upstream builds and passes its tests on the Mac. Port not started |

## Which model ("Qwen 3.8 30B 8-bit")

There is no Qwen 3.8 30B. The closest real models, checked on Hugging Face on 2026-11-03:

| Model | Type | 4-bit | 6-bit | 8-bit | Fits 32 GB at 8-bit? |
|---|---|---|---|---|---|
| `Qwen3.6-35B-A3B` | MoE, ~3B active | 20 GB | 29 GB | 38 GB | **no** → Track B |
| `Qwen3.8-27B` | dense | 16 GB | ~22 GB | ~29 GB | **no** (no room left for macOS) |
| `Qwen3-Coder-30B-A3B-Instruct` | MoE, code-tuned, older | ~17 GB | ~25 GB | ~32 GB | **no** |

So 8-bit doesn't fit in memory for any of them. Track A runs them at 4–6 bit, which is fast and has a small quality loss that the benchmark measures. Track B is the only way to get true 8-bit on this machine: keep the shared weights in RAM and stream the ~3B active experts from SSD.

## Track A quick start (Mac)

```bash
pip install mlx-lm mcp
mlx_lm.server --model mlx-community/Qwen3.6-35B-A3B-4bit --port 8080
python scripts/local/bench.py --name mlx-qwen36-35b-a3b-4bit
claude mcp add local-llm -- python ~/kimi-k3-c-engine/scripts/local/mcp_bridge.py
```

Other MLX repos that exist: `mlx-community/Qwen3.6-35B-A3B-{5bit,6bit,8bit}`, `mlx-community/Qwen3.8-27B-{4bit,8bit}`, and `mlx-community/Qwen3-Coder-30B-A3B-Instruct-{4bit,6bit}`. GGUF for llama.cpp: `unsloth/Qwen3.6-35B-A3B-GGUF`, `unsloth/Qwen3.8-27B-GGUF`.

## Track B quick start

```bash
bash scripts/setup.sh            # upstream engine: build + weightless tests (macOS: brew install libomp)
bash scripts/setup-baseline.sh   # llama.cpp CPU baseline (needs cmake)
```

## Layout

```
docs/local-agent-plan.md     Track A: speed arithmetic, candidates, phases, go/no-go
docs/plan.md                 Track B: C engine port plan
scripts/local/bench.py       prefill + decode tok/s against mlx_lm.server / llama-server
scripts/local/mcp_bridge.py  MCP server: Claude Code delegates tasks to the local model
scripts/setup.sh             upstream engine at pinned SHA
scripts/setup-baseline.sh    llama.cpp at pinned tag, CPU-only
scripts/sync.sh              two-way sync of main with GitHub
scripts/install-autosync.sh  launchd job running sync.sh every 5 min (macOS)
results/                     one folder per run (metrics committed, raw logs ignored)
third_party/                 upstream engine + llama.cpp (gitignored)
```

## Sync

`main` on GitHub is the single source of truth. Cloud sessions pull at start and push to `main` when done. The local Mac runs `scripts/sync.sh` every 5 min (install with `bash scripts/install-autosync.sh`, remove with `--uninstall`, log at `~/Library/Logs/kimi-k3-c-engine-sync.log`).
