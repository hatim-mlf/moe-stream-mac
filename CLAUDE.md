# kimi-k3-c-engine

Goal: reproduce running Kimi K3 (2.78T MoE) on the pure-C CPU engine `kimi-k3-in-c`, measure it against its claims, and decide whether it's useful to us. Read `README.md`, then `docs/plan.md`, before doing anything.

## Rules

- Upstream code lives in `third_party/kimi-k3-in-c` (gitignored, pinned by `UPSTREAM_SHA` in `scripts/setup.sh`). Don't edit it. Put wrappers, benchmarks and notes in this repo.
- Never commit model weights, trunks, `HF_TOKEN` or raw logs. Every run writes `results/<UTC timestamp>-<name>/metrics.json` with SHA, machine, preset, prompt, `--gen` and command.
- Don't start the 1.56 TB download without explicit approval from the user and confirmed ~1.7 TB of free fast disk.
- Report numbers next to upstream's. Write "did not reproduce" plainly when that's the result.
- Cloud sessions can build and run `make test`. They can't hold the checkpoint.

## Sync (local Mac ↔ GitHub ↔ cloud sessions)

Everything goes through `main` on GitHub. The local Mac auto-syncs every 5 minutes (`scripts/sync.sh` via launchd).

- **Start of a session:** `git pull --rebase origin main`.
- **During work:** commit small and often, using conventional messages.
- **End of a session, or after each finished step:** `git pull --rebase origin main && git push origin HEAD:main`. If pushing to `main` is refused, push your branch and open a PR to `main`, then say so.
- Never force-push `main`.

## Commands

- Build + weightless tests: `bash scripts/setup.sh`
- Upstream binary: `third_party/kimi-k3-in-c/bin/k3 <model> --trunk <trunk> --preset laptop --tok <model> --prompt "..." --gen 8 --incremental`
