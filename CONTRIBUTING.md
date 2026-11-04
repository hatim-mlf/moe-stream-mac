# Contributing

Thanks for trying this. The most useful contributions:

1. **Measurements on other Macs** (M2/M3/M4, 16/24/64/128 GB). Run the chat or `engine/build/run_model` and open
   an issue with the stats line, your Mac, and what else was running.
2. **Fixes and speed-ups**, with proof that the output is unchanged.
3. **More model families** (other MoE architectures) in either engine.

## Rules for code changes

- **Correctness first.** A change to the math must keep `make -C engine test` green and the whole-model check
  passing (`engine/tests/model_check.py check`: same top token at every position as MLX float32). For the MLX
  engine, `engine/hybrid/hybrid_vs_mlx.py` must stay identical to plain MLX.
- **Measure under real conditions.** Report speed with your normal apps open, not on an emptied machine:
  the goal is an engine that lives alongside daily work. Put numbers in `results/<UTC timestamp>-<name>/metrics.json`.
- **C:** C99, `-Wall -Wextra -Werror` clean, functions under ~50 lines, no hidden allocation in hot loops.
- **Never commit** model weights, tokens or logs.

## Open problems (good first contributions)

| Problem | Where | Notes |
|---|---|---|
| Reusable read buffers in the hybrid engine | `engine/hybrid/predictive.py`, `hybrid.py` | then re-enable next-layer prediction; target ~0.6 s/token on the 122B |
| Engine slows down on long answers under memory pressure | both | macOS swaps the engine's own memory; try `F_NOCACHE` reads, smaller buffers |
| Batched prefill in the C engine | `engine/src/model.c` | today prompts go token by token |
| Other architectures | `engine/src/config.c` refuses them on purpose | gpt-oss, GLM, DeepSeek MoE, Qwen3-Next coders |
| Port the hybrid to MLX's C++ API | new | removes Python overhead and per-layer sync |
