# Brief 42: run frozen GDN layers on the inference kernel during LoRA training (Codex)

<context>
`mlx_lm/models/qwen3_5.py:183-193` (mlx-lm 0.32.2 in `.venv`) picks `use_kernel=not self.training`, so during
training every Gated DeltaNet layer runs the slow per-token ops path (`gated_delta.py:557-602`), including the
frozen layers *below the first LoRA layer*, through which no gradient flows (their inputs and parameters need
no cotangent). Measured cost: the ops forward keeps ~2.3 MiB/token per layer alive. Today
`tools/gdn_checkpoint.py` (GDN_CHECKPOINT=1) halves it; 1-layer training on v1 runs at ~17 s/step.
`mlx_lm/tuner/trainer.py:263,307` call `model.train()` every step, which resets `training` on all modules.
Read those files plus `tools/ft_lora_lowmem.py`, `tools/gdn_checkpoint.py`, `tools/test_gdn_checkpoint.py`.
</context>

<task>
Write `tools/frozen_gdn.py` with `install(model)`: after LoRA layers are applied, find the first decoder layer
with a trainable parameter; for every GDN layer below it, make its `gated_delta_update` call use the inference
kernel (`use_kernel=True`) even in training mode, robust to `model.train()` being called each step (e.g.
mark those layer objects and wrap the qwen3_5 call site / the update function so it checks the mark; do not
rely on the module `training` flag). Layers at or above the first LoRA layer keep today's behaviour (ops path,
optionally gdn_checkpoint). Refuse (raise) if the model has no trainable parameters or is not qwen3_5-family.
Wire it into `tools/ft_lora_lowmem.py` behind `FROZEN_GDN_KERNEL=1` (default 0 until the lead measures it):
it must run after mlx_lm has applied LoRA and before training starts, so hook `mlx_lm.lora.train_model` or the
trainer's `train` (mirror how ft_lora_lowmem already wraps `lora.train`) and call install(model) there.
</task>

<tests>
`tools/test_frozen_gdn.py` (pytest; the lead runs it on the Mac, you have no Metal): tiny random-init
qwen3_5_moe (`_make_tiny_moe` in tools/test_gdn_checkpoint.py, 4 layers incl. both types) with LoRA on the last
layer only: loss and LoRA gradients with install() match without it (atol 1e-3: the kernel and ops forward
differ in rounding; report the max diff); a spy shows the frozen GDN layers call the kernel path and the LoRA
layer the ops path, also after `model.train()` is called again; install() raises when nothing is trainable.
</tests>

<constraints>
- Do not modify `.venv`, `gdn_checkpoint.py`, `gdn_vjp.py`, `mem_guard.py`, `mem_preflight.py`. Named
  constants, type hints, functions < 50 lines, no silent excepts. No commit.
- Report: files, the hook point you chose and why, what you could not run.
</constraints>
