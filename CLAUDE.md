# kimi-k3-c-engine

Goal: port the pure-C CPU engine `kimi-k3-in-c` so it runs ~70B-class 8-bit models: `Qwen3.6-35B-A3B-FP8` (dev) and then `Qwen3.5-122B-A10B-FP8` (main). Benchmark it against llama.cpp Q8_0 on CPU. Read `README.md`, then `docs/plan.md`, before doing anything.

## Rules

- `third_party/` (upstream engine, llama.cpp) is read-only and gitignored. Our code goes in `engine/` (C99) and `tools/` (Python reference). Keep upstream's Apache-2.0 notice on copied files.
- Every new kernel or layer gets a fixture-gated test against the Python reference **before** it's used on real weights. Stay as strict as upstream: refuse unknown configs, and never substitute defaults.
- No model download larger than 1 GB without explicit approval from the user. The Mac has ~252 GB free, so keep one big model on disk at a time.
- Never commit weights, GGUF files, `HF_TOKEN` or raw logs. Results go in `results/<UTC timestamp>-<name>/metrics.json`.
- Report numbers next to the llama.cpp baseline, and say plainly when we're slower.
- Cloud sessions: Phases 1–2 (tiny random-init models) are fine there. Phase 3 runs on the Mac.

## Sync (local Mac ↔ GitHub ↔ cloud sessions)

Everything goes through `main` on GitHub. The local Mac auto-syncs every 5 minutes (`scripts/sync.sh` via launchd).

- **Start of a session:** `git pull --rebase origin main`.
- **During work:** commit small and often, using conventional messages.
- **End of a session, or after each finished step:** `git pull --rebase origin main && git push origin HEAD:main`. If pushing to `main` is refused, push your branch and open a PR to `main`, then say so.
- Never force-push `main`.

## Commands

- Upstream engine build + tests: `bash scripts/setup.sh` (macOS needs `brew install libomp`)
- Baseline: `bash scripts/setup-baseline.sh`, then `third_party/llama.cpp/build/bin/llama-bench -m <gguf>`
