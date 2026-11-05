# Brief 36b: memory measurement harness (Codex)

Read `docs/briefs/36-memory-model.md` first (model facts, config, measured points). This piece is the tooling for
steps 2-3. Your sandbox has no Metal: write the code and tests; the lead runs them on the Mac.

<task>
Write `tools/mem_profile.py` with four subcommands. Every subcommand that loads the real model must
`import mem_guard; mem_guard.start()` first (see `tools/mem_guard.py`), call
`mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])` like `tools/ft_lora_lowmem.py`, and
write one JSON object per measurement to `--out <file.jsonl>` (append) with the config, `mx.get_peak_memory()`,
`mx.get_active_memory()`, `mx.get_cache_memory()` and wall time.

1. `step --tokens T --layers L [--gdn-checkpoint 0|1] [--no-grad-checkpoint] [--capture path.gputrace]`:
   build the model exactly as `mlx_lm.lora` does for our config (`docs/finetune/lora.yaml`: load, freeze,
   `linear_to_lora_layers` with the yaml's keys/rank/scale on the last L layers, grad_checkpoint wrapping as
   `mlx_lm/tuner/trainer.py` does, `model.train()`, Adam lr 1e-5) and run ONE training step on one row of exactly
   T tokens. The row: real text, the first row of `data/finetune/probe2/train.jsonl` that is ≥ T tokens after
   the chat template, truncated to T, with the completion mask covering the last half of the row (routing must
   see real tokens). Loss = `tools/ft_lowmem_loss.completion_only_loss`. Record phases, each followed by
   `mx.eval` and `mx.reset_peak_memory()` before the next: `load` (after weights evaluated), `forward` (loss only,
   no grad), `value_and_grad`, `optimizer_update`. Report per-phase peak and active. `--capture` wraps the
   value_and_grad phase in `mx.metal.start_capture(path)` / `stop_capture()` (requires env
   `MTL_CAPTURE_ENABLED=1`; refuse with a clear message if it is unset).
2. `waterfall --tokens T --layers L`: same model and row, but during value_and_grad record active memory per
   layer: wrap each decoder layer's `__call__` so that, in a separate instrumented pass, the forward of layer i
   is evaluated and active memory recorded after it, and the backward contribution is measured by computing
   value_and_grad of the loss with the graph cut (stop_gradient) below layer i for i from the first LoRA layer
   down. Document in the module docstring exactly what each number means and its limits under lazy evaluation.
3. `block --kind moe|gdn|attention --layer-index i --tokens T [--lora 0|1]`: isolated micro-benchmark of one
   real-weight sub-block of decoder layer i (the MoE `mlp`, the `linear_attn`, or the `self_attn`), input
   `x = random normal [1, T, 2048]` in the model dtype: peak of `mx.value_and_grad` of `sum(block(x))` w.r.t. x
   (and LoRA params when `--lora 1`), minus the active memory before the call. Also forward-only peak.
4. `sweep --tokens 256,512,1000,2000,3000 --layers 1,2,3,4 [--repeat 1]`: run `step` for every grid point, each
   in its OWN subprocess (clean peak; a subprocess killed by mem_guard (exit 86) or by a Metal OOM is recorded as
   `{"status": "guard_stop"|"oom", ...}` and the sweep continues), smallest configs first.
   And `fit --in sweep.jsonl`: least-squares fit of `peak = W + a·L + b·T + c·L·T` on the completed points
   (numpy is in the venv), printing coefficients in GiB, GiB per layer, MiB per token, MiB per layer·token,
   residuals, R².
</task>

<tests>
`tools/test_mem_profile.py` (pytest; the lead runs it on the Mac): with a tiny random-init qwen3_5_moe (reuse
the tiny config in `tools/test_gdn_checkpoint.py::_make_tiny_moe`): `step` logic returns all four phases with
non-negative numbers and the loss equals `completion_only_loss` computed directly; row builder returns exactly T
tokens and a mask over the last half, refuses T larger than every row; `fit` recovers known coefficients from
synthetic points within 1e-6; sweep marks a child exiting 86 as `guard_stop` (fake child). Structure the code so
these run without the real model (inject the model / row source).
</tests>

<constraints>
- Do not modify `.venv`, `mem_guard.py`, `ft_lora_lowmem.py`, `ft_lowmem_loss.py`, `gdn_checkpoint.py` or other
  tools; import them. Mirror mlx_lm's own LoRA setup by calling its functions, not re-implementing them, and cite
  the file:line you mirror.
- Named constants, type hints, functions < 50 lines, no silent excepts. No commit.
- Report: files, what you could and could not run, anything unverified.
</constraints>
