# moe-stream-mac

Two engines that run Qwen 3.5/3.6 MoE models bigger than RAM by streaming experts from SSD: the C engine
(`engine/src`, docs/C_ENGINE.md) and the MLX hybrid engine (`engine/hybrid`, docs/MLX_ENGINE.md), both driven by
`engine/cli/qchat.py`. Read `README.md` and `docs/REPORT.md` first. Old plans live in `docs/history/`.

- Every math change must keep `make -C engine test` green and the whole-model check matching MLX
  (`engine/tests/model_check.py`, `engine/hybrid/hybrid_vs_mlx.py`).
- Measure with the user's normal apps open (Xcode etc.); never ask them to close apps for cleaner numbers.

## Working model (rule, set by the user on 2026-11-04)

- **Claude is the orchestrator:** writes specs, interfaces, tests and skeletons; reviews every diff; runs the
  verification and measurements; decides. Claude does not write implementation code it can delegate.
- **Implementation goes to the helpers:** the local helper model, KAT-Coder by default (`tools/delegate.py --test CMD`, needs `scripts/local/serve-mlx.sh`; `HELPER=qwen-rft` switches back)
  for focused, well-specified pieces; **Codex** (GPT via the codex CLI) for larger or multi-file pieces and for
  independent review. Commit messages via `tools/qwen_commit.py`.
- Log every delegation in `docs/delegation-log.md` (who, attempts, what went wrong).

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
