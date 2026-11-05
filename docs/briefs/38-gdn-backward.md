# Brief 38: a GDN training backward that bounds memory (Codex)

<context>
LoRA training of Qwen3.6-35B-A3B (MLX 0.32.4, mlx-lm 0.32.2 in `.venv`) is memory-bound by the Gated DeltaNet
(GDN) backward: `mlx_lm/models/qwen3_5.py:183-193` calls `gated_delta_update(..., use_kernel=not self.training)`,
so training runs `gated_delta_ops` (`mlx_lm/models/gated_delta.py:385-427` step, `:557-602` loop), and autodiff
keeps ~4 state-sized arrays per token: state [B, Hv=32, Dv=128, Dk=128] fp32 = 2 MiB, so ~7-8 MiB/token
(measured; `docs/finetune/MEMORY_MODEL.md` §2-4). `tools/gdn_checkpoint.py` (chunked `mx.checkpoint`) does not
help: `results/20261105T143800Z-memory-model/repro/ckpt_toy2.py` shows that mx.checkpoint stops bounding memory
once the per-step inputs (q, k, v) need gradients. Read those three files and `tools/test_gdn_checkpoint.py`.
Target: GDN backward memory O(T/C + C) states instead of O(T) (C = chunk), e.g. ~0.5 GiB at T = 1000.
</context>

<task>
Write `tools/gdn_vjp.py` with `gated_delta_chunked(q, k, v, g, beta, state, mask=None, chunk=64)`: same
signature, dtypes and return (y, final_state) as `gated_delta_ops` (GQA repeat of q/k heads, g of shape [B,T,Hv]
or [B,T,Hv,Dk], mask semantics), built with `mx.custom_function` so autodiff never sees the per-step graph:

1. Forward: run the recurrence chunk by chunk; keep only the chunk-boundary states (stop_gradient copies) as
   saved values; return y and the final state.
2. Custom vjp: derive the backward of the gated delta rule by hand (write the per-step adjoint equations in the
   module docstring: for S_t = S_{t-1}·decay_t + k_t ⊗ δ_t, δ_t = β_t (v_t − (S_{t-1}·decay_t) k_t),
   y_t = S_t q_t; give dq, dk, dv, dβ, dg, dS_{t-1} from dy_t and dS_t). Walk the chunks in reverse; in each,
   recompute the chunk's states from its boundary state and run the reverse recurrence. Return cotangents for
   q, k, v, g, beta, state (and None for mask).
3. Order of evaluation matters in lazy MLX: whole-graph evaluation may keep every chunk's recomputed states
   alive at once (that is the mx.checkpoint failure). Make the dependency chain force chunk-at-a-time
   lifetimes. Option (a): each chunk's recompute takes the incoming state cotangent from the later chunk as a
   (zero-weight) data dependency, so it cannot run early. Option (b), if (a) is not enough: implement the
   per-chunk backward as one `mx.fast.metal_kernel` (states recomputed inside the kernel, never materialised),
   modelled on the upstream forward kernel `gated_delta.py:429-555`. Provide (a); add (b) behind
   `impl="kernel"` if you can make it correct against (a).
4. `install()` like `tools/gdn_checkpoint.py`: patch `qwen3_5.gated_delta_update` so the training path
   (use_kernel=False) uses `gated_delta_chunked` and the inference path is untouched; `tools/ft_lora_lowmem.py`
   gets an env flag `GDN_VJP=1` (default 0) calling it before `lora.main()`, next to `GDN_CHECKPOINT`.
5. `tools/gdn_vjp_mem.py T [--impl ops|chunked|kernel]`: memory probe (real GDN shapes, random inputs, grads
   w.r.t. q, k, v, g, beta) printing backward peak MiB and MiB/token, like the repro's `gdn_iso.py`.
</task>

<tests>
`tools/test_gdn_vjp.py` (pytest; the lead runs it on the Mac, your sandbox has no Metal): forward y and final
state match `gated_delta_ops` (fp32, atol 1e-5) for T in {1, 63, 64, 65, 200}, chunk in {16, 64}, with/without
mask, g 3-D and 4-D; all six cotangents match `mx.vjp` of `gated_delta_ops` (atol 1e-4); install() leaves
use_kernel=True untouched (spy); tiny qwen3_5_moe (reuse `_make_tiny_moe` from tools/test_gdn_checkpoint.py):
loss and LoRA grads identical with install() vs without (atol 1e-4).
</tests>

<constraints>
- Do not modify `.venv`, `gdn_checkpoint.py` or other tools except the flag in ft_lora_lowmem.py. Named
  constants, type hints, functions < 50 lines, no silent excepts. No commit.
- Report: files, the adjoint equations, what you could not run, anything unverified.
</constraints>
