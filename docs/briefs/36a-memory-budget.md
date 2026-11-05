# Brief 36a: first-principles memory budget (Codex)

Read `docs/briefs/36-memory-model.md` first: it holds the model facts, training config and every measured point.
This piece is step 1 of that task: the budget from source, as code plus a written derivation. No Metal is needed.

<task>
Derive, from source and not from the measurements, the memory of one `mlx_lm.lora` training step (batch 1) as a
function of T (tokens in the row) and the set of LoRA layers (types GDN / attention per brief 36), and write it as:

1. `tools/mem_budget.py` (stdlib only):
   - `load_shapes(config_path) -> Shapes` (frozen dataclass of every size you use; refuse missing keys).
   - `budget(shapes, tokens: int, lora_layers: int, *, grad_checkpoint=True, rank=8) -> dict[str, int]`: one
     entry per memory term in bytes (weights resident, LoRA params, LoRA grads, Adam m/v, forward activations kept
     by autodiff in frozen and in LoRA layers, the per-layer recompute under grad_checkpoint, backward temporaries
     of each op that appears in the backward path, logits / loss, and anything else you find). Plus
     `peak_bytes(...)`: your model of which terms are alive at the same time at the peak (say which moment that is).
   - CLI: `python3 tools/mem_budget.py --config <config.json> --tokens 1000 --layers 2` prints the term table
     in GiB and the predicted peak; `--points` prints predictions for every measured point in brief 36 next to
     the measured value and the % gap.
2. `docs/finetune/MEMORY_MODEL.md`, section "## 1. Budget from source": one row per term (formula, bytes at
   T=1000 L=2, the source file:line it comes from), then the prediction-vs-measurement table from `--points`.
   Every gap > 10% goes in a list "Open questions" with the term you suspect; do not tune constants to close
   gaps, and do not claim a cause you have not traced in source.

Source to read (cite file:line for every term):
- mlx-lm 0.32.2: `.venv/lib/python3.14/site-packages/mlx_lm/` — `models/qwen3_5.py`, `models/qwen3_5_moe.py`,
  `models/switch_layers.py` (SwitchGLU, `gather_qmm`, index sorting), `models/gated_delta.py` (ops path used in
  training: `use_kernel=not self.training`), `tuner/trainer.py` (grad_checkpoint wrapping, step, loss),
  `tuner/lora.py`, `lora.py`, `tools/ft_lowmem_loss.py` (our completion-only loss), `tools/gdn_checkpoint.py`.
- MLX C++ 0.32.4: `/Users/hatim/src/mlx-0.32.4/mlx/primitives.cpp` (`QuantizedMatmul::vjp` ~3596,
  `GatherQMM::vjp` ~3782, Gather/Scatter, Softmax, matmul vjps), `mlx/ops.cpp` (`gather_qmm`, `quantized_matmul`
  shapes), `mlx/backend/metal/` only where a kernel allocates temporaries, and how `mx.checkpoint` /
  `vjp` decide what is kept (`mlx/transforms.cpp`).
Key questions the budget must answer explicitly: what does the input-gradient of a frozen 4-bit `gather_qmm`
allocate (does anything get dequantized, and at what size: per token × top-k, or per expert)? What does
autodiff keep for the 40 − L frozen layers in front of the first LoRA layer? What does grad_checkpoint keep per
layer and what does recompute allocate at the backward of one layer?
</task>

<tests>
`tools/test_mem_budget.py` (pytest, pure Python, you can run it): shape loading refuses a config with a missing
key; each term scales as its formula says (e.g. doubling T doubles a per-token term, LoRA params independent of
T); weights term matches the real safetensors byte total within 1% when the model dir exists (skip otherwise).
Run: `python3 -m pytest -q tools/test_mem_budget.py` from the repo root.
</tests>

<constraints>
- Read-only everywhere except the three files above. Do not modify `.venv`, `/Users/hatim/src`, other tools.
- Named constants, type hints, functions < 50 lines, no silent excepts. No commit.
- Report: files, test result, the 3 largest terms at T=1000 L=2, every open question, anything unverified.
</constraints>
