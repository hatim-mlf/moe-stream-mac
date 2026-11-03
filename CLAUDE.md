# kimi-k3-c-engine

Goal: run a strong ~30B-class Qwen model locally on the M1 Max (32 GB) and let Claude Code delegate work to it. Read `README.md` first.

- **Track A** (`docs/local-agent-plan.md`): MLX / llama.cpp serving at 4–6 bit, `scripts/local/bench.py`, `scripts/local/mcp_bridge.py`.
- **Track B** (`docs/plan.md`): port the pure-C engine `kimi-k3-in-c` to `qwen3_5_moe`, so that 8-bit `Qwen3.6-35B-A3B` (38 GB, bigger than RAM) streams experts from SSD. It has to beat Track A's 8-bit numbers to be worth it.
- There is no Qwen 3.8 30B. Use the model names in the README table, which were checked on Hugging Face.

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
