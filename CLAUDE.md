# c2c-eval

Goal: verify the C2C (Cache-to-Cache) paper's claims and decide whether it fits our agent stack. Read `README.md`, then `docs/plan.md`, before doing anything.

## Rules

- Upstream code lives in `third_party/C2C` (gitignored, pinned by `UPSTREAM_SHA` in `scripts/setup.sh`). Don't edit it. Put wrappers and evaluation glue in `scripts/`.
- Every run writes `results/<UTC timestamp>-<name>/metrics.json` with the upstream SHA, hardware, dtype, command and seed. Commit metrics and never commit raw generations or model weights.
- Use the same prompts, decoding settings and seed in every mode (receiver alone, sharer alone, T2T, C2C). Any comparison that breaks this is invalid.
- Report numbers with CIs and next to the paper's numbers. Write "did not reproduce" plainly if that's the result.
- Cloud sessions have no GPU and may block Hugging Face downloads. Phase 0 can run on CPU; Phases 1–2 need a CUDA box.

## Commands

- Setup: `bash scripts/setup.sh && source .venv/bin/activate`
- Smoke test: `python scripts/smoke_test.py --device cpu`
- Upstream eval: `python third_party/C2C/script/evaluation/unified_evaluator.py --config <yaml>`
