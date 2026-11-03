# Plan

Upstream is pinned in `scripts/setup.sh` (`UPSTREAM_SHA`). Every result folder records the SHA, machine (CPU, cores, RAM, disk model), preset, prompt, `--gen` count and the full command.

## Phase 0: Build + weightless tests (cloud session or Mac, minutes)

- `bash scripts/setup.sh`, which runs `make -j && make test`.
- Pass: `ALL WEIGHTLESS TESTS PASSED` on both Linux (cloud) and macOS arm64 (NEON).
- Also read the code: the 7 C files, the expert-streaming design, the MXFP4 kernels. Write notes in `docs/notes.md`.

## Phase 1: Hardware decision (no download yet)

Choose one:

| Option | Cost | Notes |
|---|---|---|
| Mac M1 Max + external 2 TB NVMe (Thunderbolt) | ~€150–250 one-off | 32 GB RAM means roughly the 32 GB row (~24 s/token upstream). USB/TB disk speed dominates |
| Rented CPU server, ≥2 TB NVMe, 128 GB+ RAM | hourly; delete after | Closest to the upstream 5.6 s/token row. Download time (1.56 TB) is billed too |

Gate: don't start the 1.56 TB download until the disk is in place and Phase 0 passes.

## Phase 2: Download, pack, first tokens

- `export HF_TOKEN=...` (environment only, never committed), then `./scripts/download-model.sh <dir>`, which is resumable.
- The verify step must print `RESULT : byte-exact match`. Treat any `FAIL` as a stop, because partial downloads produce wrong tokens silently.
- `./scripts/pack-trunk.sh <model> <trunk>`.
- Run the upstream demo prompts ("The capital of France is", "def fibonacci(n):") with `--gen 8` and `--gen 28`.

## Phase 3: Measure

- K1: peak RSS per preset (`laptop`, then larger ones that fit).
- K2: s/token, as the mean of 3 runs after a warm-up, with disk read throughput logged.
- K3: `diff` the generated text across presets. It must be byte-identical.
- Compare with the hosted K3 API: latency and cost per 1K output tokens.

## Go/no-go

- **Keep going** (chat mode, longer prompts, an agent experiment) if K1–K3 reproduce and s/token is tolerable for offline or batch use.
- **Stop** if the claims don't reproduce, or if the measurements confirm it's only a curiosity for us. Record the findings and archive the repo.
