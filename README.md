# kimi-k3-c-engine

Hands-on test of running **Kimi K3** (Moonshot AI, 2.78T-parameter MoE) on a pure-C CPU inference engine: [FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (C99, no BLAS/framework/GPU, Apache-2.0, not affiliated with Moonshot).

## Claims to verify

| # | Claim (upstream README) | How we check |
|---|---|---|
| K1 | Runs in 8.24 GB peak RSS (`--preset laptop`) | `PEAK RSS` line + `/usr/bin/time -l` |
| K2 | ~26.5 s/token at 8 GB, ~5.6 s/token at 128 GB+ (124 cores, fast NVMe) | s/token on our hardware, same prompts |
| K3 | Output is byte-identical across memory presets | diff generations across presets |
| K4 | Weightless test suite proves the engine matches its reference | `make test` |

## Is it useful to us?

Likely as a **learning and benchmarking exercise, not a production path**. At 5–30 s per token, one short answer takes minutes, and the hosted K3 API is faster and cheaper. The value is in understanding MoE expert streaming, quantisation (MXFP4) and memory/disk trade-offs, plus having a fully offline frontier model if that's ever required.

## Hardware reality

- Disk: **~1.7 TB free** (1.56 TB checkpoint + 109 GB packed trunk), ideally fast NVMe.
- This Mac (M1 Max, 32 GB, ~252 GB free) can build and run `make test`, but needs an external ≥2 TB NVMe SSD for the real model.
- Cloud sessions: fine for building, tests and code reading. They can't hold the checkpoint.

## Layout

```
docs/plan.md               phased plan + go/no-go criteria
scripts/setup.sh           clones upstream at a pinned commit, builds, runs weightless tests
scripts/sync.sh            two-way sync of main with GitHub
scripts/install-autosync.sh  launchd job running sync.sh every 5 min (macOS)
results/                   one folder per run (metrics committed, raw logs ignored)
third_party/kimi-k3-in-c   upstream code (gitignored, created by setup.sh)
```

## Sync

`main` on GitHub is the single source of truth. Cloud sessions pull at start and push to `main` when done. The local Mac runs `scripts/sync.sh` every 5 min (install with `bash scripts/install-autosync.sh`, remove with `--uninstall`, log at `~/Library/Logs/kimi-k3-c-engine-sync.log`).

## Quick start

```bash
bash scripts/setup.sh   # clone pinned upstream, make -j, make test (no weights, no network after clone)
```
