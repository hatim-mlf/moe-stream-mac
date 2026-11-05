# Brief 36c: independent review of the memory budget (Codex, read-only)

<task>
`tools/mem_budget.py` claims to predict the MLX memory of one `mlx_lm.lora` training step (batch 1, LoRA on the
last L layers, grad_checkpoint on) for the model whose config is
`/Users/hatim/models/Qwen3.6-35B-A3B-4bit/config.json`, training config `docs/finetune/lora.yaml`, entry point
`tools/ft_lora_lowmem.py` (completion-only loss `tools/ft_lowmem_loss.py`, optional `tools/gdn_checkpoint.py`).
Review every term in `mem_budget.py` against the source, independently. Do NOT read `docs/finetune/MEMORY_MODEL.md`,
`results/`, `docs/briefs/36*.md` or any notes: judge the formulas from code only.

Source: mlx-lm 0.32.2 at `.venv/lib/python3.14/site-packages/mlx_lm/` (models/qwen3_5.py, qwen3_5_moe.py,
switch_layers.py, gated_delta.py, tuner/trainer.py, tuner/lora.py, lora.py); MLX 0.32.4 C++ at
`/Users/hatim/src/mlx-0.32.4` (mlx/primitives.cpp vjps, mlx/transforms.cpp vjp/checkpoint/eval,
mlx/backend/metal/allocator.cpp).

For each term: is the formula right (shape, dtype, count), is it really alive at the claimed peak moment, cite the
source line. Then list terms the budget is MISSING, and terms it double-counts. Pay special attention to:
(1) the GDN training path (`use_kernel=not self.training` → ops loop) and how many state-sized arrays per token
the backward keeps, with and without `gdn_checkpoint.py`; (2) whether peak should be a sum over LoRA layers or a
max (grad_checkpoint); (3) what the allocator cache adds to process memory and its default limit.
</task>

<constraints>
Read-only: change no file. Report: a table term → verdict (correct / wrong + fix / unsure) → source file:line;
missing terms; your own per-token GDN formula; the 3 changes that would most improve accuracy. ≤ 1,200 words.
</constraints>
