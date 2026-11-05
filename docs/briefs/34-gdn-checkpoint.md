<task>
LoRA training of our Qwen3.6-35B-A3B (MLX 4-bit, mlx-lm 0.32.2 in .venv, model_type qwen3_5_moe; 40 layers, every
4th is full attention, the other 30 are Gated DeltaNet linear attention: 32 value heads, 16 key heads, head dims
128/128) runs out of GPU memory as soon as a Gated DeltaNet layer is in the backward path.
Cause (verified by reading the source): `mlx_lm/models/qwen3_5.py:183-193` calls
`gated_delta_update(..., use_kernel=not self.training)`, so in training it uses `gated_delta_ops`
(`mlx_lm/models/gated_delta.py:385` `_gated_delta_step_ops`, ~557 `gated_delta_ops`): a Python loop over time
steps whose autodiff graph retains every per-token recurrent state [B, Hv, Dv, Dk] fp32 = 2 MB per token per layer.
Measured on the Mac (batch 1): 1 adapted layer (layer 39, full attention) 19.5 GiB at 470 tokens; 4 layers 25.1 GiB;
2 layers OOM (> 28 GiB) at 1.2-3k tokens.

Write `tools/gdn_checkpoint.py` with `gated_delta_ops_checkpointed(q, k, v, g, beta, state, mask=None, chunk=64)`:
numerically identical to `gated_delta_ops` (same signature/return (out, final_state), same dtypes, same mask
semantics), but it splits the time axis into chunks and runs each chunk's recurrence under `mx.checkpoint`, so the
backward keeps only chunk-boundary states and recomputes one chunk at a time (memory ~ (T/chunk + chunk) states
instead of T). Pass integer/non-differentiable arguments by closure, not as checkpoint positional args
(mx.checkpoint differentiates every positional argument; e.g. the mask). Plus `install()` that monkeypatches the
name qwen3_5.py actually resolves (it imports `gated_delta_update` from .gated_delta: patch so the training path,
use_kernel=False, uses the checkpointed ops, and the inference path, use_kernel=True, is untouched). Read both files
to get every detail right (state layout, GQA repeat of k/q heads, g/beta shapes, mask handling, dtype casts).
Then `tools/ft_lora_lowmem.py` (exists; read it): call `gdn_checkpoint.install()` before `lora.main()`, behind an
env flag `GDN_CHECKPOINT=1` (default on) so the lead can A/B it. Do not change its other behaviour.
</task>

<tests>
`tools/test_gdn_checkpoint.py` (pytest; the lead runs it on the Mac, you cannot run MLX):
- random q,k,v,g,beta,state with the real head counts but small T (T in {1, 63, 64, 65, 200}) and chunk in
  {16, 64}, with and without a mask: forward outputs and final state match `gated_delta_ops` (atol 1e-5 fp32),
  and gradients w.r.t. q,k,v,g,beta,state via mx.vjp / value_and_grad of a scalar function match (atol 1e-4).
- a tiny random-init qwen3_5_moe model (shrink the real config: 4 layers incl. both layer types, hidden 64,
  experts 4, top-k 2, vocab 1000) in training mode with LoRA on all 4 layers: loss and LoRA grads identical with
  install() vs without (atol 1e-4).
- install() leaves the use_kernel=True path unchanged (spy).
</tests>

<constraints>
- Do not modify .venv or any other tool. Named constants, type hints, functions < 50 lines, no silent excepts.
- Do not commit. Report: files, the source lines you mirrored, and anything unverified.
</constraints>
