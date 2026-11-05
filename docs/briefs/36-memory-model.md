# Brief 36: quantitative memory model of one LoRA training step (master task)

Set by the user on 2026-11-05 after probeF (kernel panic), probeH (mem_guard stop) and the GDN-checkpoint A/B
(no effect). Claude orchestrates; pieces go to Codex (36a budget, 36b harness); Claude runs every Metal
measurement on the Mac and decides.

## The task (user's prompt, verbatim)

> Stop probing and guessing. Build a quantitative memory model of one `mlx_lm.lora` training step for our
> Qwen3.6-35B-A3B-4bit on this 32 GB M1 Max. It must explain the measured peaks so far and predict new ones.
>
> 1. **Budget from first principles.** From `config.json` and the MLX / mlx-lm source (read the vjp of
>    `GatherQMM`, `QuantizedMatmul`, SwitchGLU, the GDN ops path, attention, and the trainer's
>    `grad_checkpoint`), write every memory term as a formula in T (tokens) and L (LoRA layers, with their
>    type: GDN or attention). Cover weights, LoRA and Adam state, saved activations, backward temporaries,
>    recomputation and logits. Predict the existing points: probeA, probeC, train2, probeI/J, probeH.
> 2. **Measure, don't infer.** Take one Xcode GPU capture of a 2-layer step at ~1k tokens, plus a per-layer and
>    per-phase active-memory waterfall. Run isolated forward+backward micro-benchmarks of one MoE, one GDN and
>    one attention block with real weights.
> 3. **Fit the law.** Sweep T ∈ {256, 512, 1k, 2k, 3k} × L ∈ {1, 2, 3, 4}, 1 iteration each, all under
>    mem_guard. Fit `peak = W + a·L + b·T + c·L·T` and name the term behind each coefficient. Report every gap of
>    more than 10% between the budget and the measurement as an open question, not a conclusion.
> 4. **Prove it.** Before running them, predict two untested configs to within 5%. Then have Codex check the
>    formulas against the source without seeing our conclusions. Any fix that follows must pass the A/B on
>    identical rows plus the loss/gradient parity tests.
> 5. **Deliver.** A table of which configs fit at the 29.5 GiB guard limit and why, a preflight check that
>    refuses over-budget configs, and a recommendation for the ~250-row run. The recommendation is judged by
>    heldout-v2 against the base model with a paired CI. Write it to `docs/finetune/MEMORY_MODEL.md`, with raw
>    numbers in `results/<UTC>-memory-model/metrics.json`.

## Facts the pieces share

- Model `/Users/hatim/models/Qwen3.6-35B-A3B-4bit` (19 GB on disk), mlx 0.32.4, mlx-lm 0.32.2 (`.venv`).
  MLX C++ source at the same tag: `/Users/hatim/src/mlx-0.32.4` (read-only).
- text_config: hidden 2048, 40 layers, `full_attention_interval` 4 (layers 3, 7, …, 39 full attention; the
  other 30 Gated DeltaNet), 256 experts, top-k 8, moe_intermediate 512, shared_expert_intermediate 512,
  vocab 248,320, attention 16 q heads / 2 kv heads / head_dim 256, GDN 32 value heads / 16 key heads / dims
  128/128, untied embeddings. Quantization affine 4-bit group 64; router `mlp.gate` and `shared_expert_gate`
  8-bit.
- Training config `docs/finetune/lora.yaml`: batch 1, rank 8, scale 20, `grad_checkpoint: true`,
  `mask_prompt: true`, Adam lr 1e-5. LoRA keys: `linear_attn.{in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,
  out_proj}`, `self_attn.{q,k,v,o}_proj`, `mlp.shared_expert.{gate,up,down}_proj`, `mlp.shared_expert_gate`.
  The routed experts (`switch_mlp`) are frozen, but the backward goes through them for input gradients.
  `--num-layers L` adapts the last L layers (L=1: layer 39, attention; L=2: 38 GDN + 39; L=4: 36-39).
- Entry point `tools/ft_lora_lowmem.py` (completion-only loss, `GDN_CHECKPOINT` flag, mem_guard).
- mem_guard stops a run at 29.5 GiB system wired (total 32 − 2.5 headroom); `iogpu.wired_limit_mb` = 31744.

## Measured points so far (MLX `mx.get_peak_memory`, batch 1)

| Run | L | Rows (tokens) | Result |
|---|---|---|---|
| probeA | 1 | 470 | 19.48 GiB |
| probe4 | 4 | 470-471 | 25.14 GiB |
| train2 | 2 | ≤ 600, 66 iters | 24.46 GiB |
| probeI / probeJ | 2 | ≤ 600 (night1), 6 iters | 22.532 (GDN ckpt on) / 22.528 (off) GiB |
| probeC / run3 / probeE | 1 | 1.2k-3k | 24.19 / 24.18 / 24.177 GiB |
| train1 | 4 | ≤ 600 | OOM at step 12 (28 GiB limit) |
| probeB / probeG | 2 | 1.2k-3k | OOM on step 1 (28 GiB / default limit; footprint 27.5 GB) |
| probeH | 2 | 1.2k-3k | mem_guard stop at 29.93 GiB system wired, footprint 30.18 GB |
| probeF | 2 | 1.2k-3k | kernel panic, process resident 31.26 GiB |

Disproved: completion-only chunked loss (no saving), per-token GDN state checkpoint (no saving, path verified
called 62 times).
