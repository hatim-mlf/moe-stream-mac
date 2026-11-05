# Memory model

## 1. Budget from source

### Scope and symbols

[`tools/mem_budget.py`](../../tools/mem_budget.py) exposes `load_shapes`, `budget`, `peak_bytes`, and the CLI. Allocation formulas live in [`tools/mem_budget_terms.py`](../../tools/mem_budget_terms.py). It models one batch-one `mlx_lm.lora` step for the local Qwen3.6-35B-A3B-4bit model, rank 8 LoRA on the last `L` layers, Adam, layer gradient checkpointing, completion-only loss in 512-position chunks, and the project GDN wrapper with chunk 64. `--points` prints the historical Brief 36 points, the new trainer sweep, isolated blocks, and held-out points. `--assume-working-gdn-checkpoint` selects a counterfactual GDN checkpoint formula; the default remains the measured `4T` envelope.

`T` is the number of model input positions; training rows contain `T+1` IDs for next-token targets. In the trainer, a row of `N` IDs is padded to `1 + 32*ceil(N/32)` and truncated at `max_seq_length`, so a row of 1000 IDs is modeled with `T=1024` when the maximum is 10240 (`tuner/trainer.py:158-172`). The final `L` entries in `text_config.layer_types` receive LoRA; at `L=2`, these are layer 38 GDN and layer 39 full attention. Model dimensions are from the local config and [Brief 36](../briefs/36-memory-model.md:33-46). Let `B=1`, `H=2048`, `V=248320`, `d=2` BF16 bytes, `E=256`, `k=8`, `I=S=512`, `Q=16`, `K_v=2`, `D=256`, `V_g=H_vD_v=4096`, `C_g=2K+V_g`, `G=3H+2K+5V_g+2H_v+2C_g+2H_vD_k`, `r=8`, `R=BTk`, and `M=BH_vD_vD_k*4=2,097,152` bytes per FP32 GDN state. `P(L,r)=r*sum_j(input_j+output_j)` across configured LoRA targets.

The inventory includes diagnostics and mutually exclusive arrays. It is not a sum. At a checkpointed backward peak, the model adds weights, trainable/optimizer state, retained hidden boundaries, and accumulated LoRA gradients, then takes the **maximum** of the GDN/attention scratch and MoE/router scratch for each replayed layer. It then takes the outer maximum across adapted layers, loss backward, and optimizer update. Saved hidden boundaries are summed over the adapted segment: `(L+1)*B*T*H*d` bytes when `L>0`. The table below is the inventory at `T=1000,L=2`; its peak uses the GDN branch of layer 38.

### Term inventory at `T=1000,L=2`

Source paths are relative to the repository unless prefixed `mlx/`, which means `/Users/hatim/src/mlx-0.32.4/mlx/`, `mlx_py/`, which means `.venv/lib/python3.14/site-packages/mlx/`, or `mlx_lm/`, which means `.venv/lib/python3.14/site-packages/mlx_lm/`. Bytes are decimal integers. Cache cap, current cache occupancy, safetensors payload, dropped weights, and OS reserve are diagnostic or wired-memory terms; do not add them all to the active peak.

| Term | Formula | Bytes | Source file:line |
|---|---|---:|---|
| `weights_resident` | safetensors payload after model sanitization | 19,508,787,456 | `mlx_lm/utils.py:449-456,493-495,598-603`; `mlx_lm/models/qwen3_5_moe.py:23-34`; `mlx_lm/models/qwen3_5.py:350-357` |
| `weights_safetensors_payload` | `sum(data_offsets.end-data_offsets.start)` over model shards | 20,401,929,952 | `mlx_lm/utils.py:449-456` |
| `weights_dropped_by_sanitize` | visual tensors removed from this archive | 893,142,496 | `mlx_lm/models/qwen3_5_moe.py:23-34`; `mlx_lm/models/qwen3_5.py:350-357` |
| `allocator_cache_cap` | `min(1.5*26.8,0.95*32) GiB` (fallback values) | 32,641,751,449 | `mlx/backend/metal/allocator.cpp:58-65,81-85`; measured device facts `docs/briefs/36-memory-model.md:33-46` |
| `allocator_gc_threshold` | `min(0.95*26.8,cache_cap) GiB` | 27,337,466,838 | `mlx/backend/metal/allocator.cpp:58-65,87-93` |
| `allocator_cache_occupancy_now` | `mx.get_cache_memory()` at the time the budget is queried; 0 in this run | 0 | `mlx/backend/metal/allocator.cpp:128-138,183-200`; `mlx/memory.h:44` |
| `trainer_cache_clear_threshold` | configured threshold, `0` bytes | 0 | `mlx_lm/tuner/trainer.py:80-85`; `mlx_lm/lora.py:72-76` |
| `trainer_cache_occupancy_after_step` | `0` after `mx.eval` when cache occupancy is positive | 0 | `mlx_lm/tuner/trainer.py:22-24,331-332` |
| `system_os_wired_reserve` | measured residual, approximately `1.7 GiB` | 1,825,361,100 | `tools/mem_guard.py:21-43`; residual estimate `results/20261105T143800Z-memory-model/NOTES.md:29-30` |
| `lora_parameters` | `P(L,r)*4` FP32 bytes | 2,164,800 | `docs/finetune/lora.yaml:13-30`; `mlx_lm/tuner/lora.py:67-98` |
| `lora_gradients` | `P(L,r)*4` | 2,164,800 | `mlx_py/nn/utils.py:12-38`; `mlx/transforms.cpp:451-495,503-557`; `mlx_lm/tuner/trainer.py:239-259` |
| `adam_m` | `P(L,r)*4` | 2,164,800 | `mlx_lm/tuner/trainer.py:258`; `mlx_py/optimizers/optimizers.py:521-539` |
| `adam_v` | `P(L,r)*4` | 2,164,800 | `mlx_lm/tuner/trainer.py:258`; `mlx_py/optimizers/optimizers.py:521-540` |
| `adam_update_temporaries` | `3*P(L,r)*4` for update-sized `m`, `v`, and parameter results | 6,494,400 | `mlx_py/optimizers/optimizers.py:535-549`; overlap is an envelope, not an exact allocator trace |
| `checkpoint_saved_frozen_prefix_activations` | 0 internal prefix activations; first LoRA layer input is counted below | 0 | `mlx_lm/lora.py:233-252`; `mlx_py/nn/utils.py:12-38`; `mlx/transforms.cpp:451-495` |
| `checkpoint_saved_lora_segment_input` | `B*T*H*d` | 4,096,000 | `mlx_lm/tuner/trainer.py:27-40`; `mlx/transforms.cpp:1033-1045,1119-1129` |
| `checkpoint_saved_lora_layer_inputs` | `(L-1)*B*T*H*d` | 4,096,000 | `mlx_lm/tuner/trainer.py:27-40`; `mlx/transforms.cpp:1033-1045,1119-1129` |
| `checkpoint_final_layer_output` | `B*T*H*d` | 4,096,000 | `mlx_lm/models/qwen3_5.py:294-315`; `mlx/transforms.cpp:1033-1045,1119-1129` |
| `final_norm_output` | `B*T*H*d` | 4,096,000 | `mlx_lm/models/qwen3_5.py:309-315`; `tools/ft_lowmem_loss.py:55-69` |
| `loss_full_sequence_hidden_gradient` | `B*T*H*d`, assembled from the chunk-slice cotangents | 4,096,000 | `tools/ft_lowmem_loss.py:40-47,55-70`; `mlx/transforms.cpp:503-557` |
| `noncheckpoint_saved_frozen_activations` | 0 internal frozen-prefix activations | 0 | `mlx_lm/lora.py:233-252`; `mlx_py/nn/utils.py:12-38`; `mlx/transforms.cpp:451-495` |
| `noncheckpoint_saved_lora_activations` | 0 with layer checkpointing; otherwise sum adapted hidden, projection, MoE, and rank activations | 0 | `mlx_lm/models/qwen3_5.py:88-201,228-240`; `mlx_lm/models/qwen3_next.py:74-151,299-346`; `mlx_lm/tuner/lora.py:86-98` |
| `gdn_state_bytes_per_step` | `M=B*H_v*D_v*D_k*4` | 2,097,152 | `mlx_lm/models/gated_delta.py:397-405,582-584` |
| `gdn_checkpoint_boundary_states` | `(ceil(T/64)+1)*M` with the project GDN wrapper enabled | 35,651,584 | `tools/gdn_checkpoint.py:10,53-95`; `mlx/transforms.cpp:1033-1045,1119-1129` |
| `gdn_step_vjp_state_arrays` | measured model `4*T*M` | 8,388,608,000 | `mlx_lm/models/gated_delta.py:408-425,557-602`; `mlx/primitives.cpp:3248-3258,3301-3328,4109-4118` |
| `gdn_step_vjp_vector_arrays` | `4*B*T*V_g*4` FP32 bytes | 65,536,000 | `mlx_lm/models/gated_delta.py:408-421`; `mlx/primitives.cpp:3248-3258,4109-4118` |
| `gdn_mask_where_vjp` | 0 because training passes `mask=None`; a non-null mask adds two state-shaped branches | 0 | `mlx_lm/models/base.py:58-61`; `mlx_lm/models/qwen3_5.py:183-194`; `mlx_lm/models/gated_delta.py:423-425`; `mlx/primitives.cpp:3301-3328` |
| `batch_targets_and_completion_mask` | `B*((T+1)*4 + 2*4 + T*(4+1))` | 9,012 | `mlx_lm/tuner/trainer.py:88-101,158-172`; `tools/ft_lowmem_loss.py:55-71` |
| `logits_chunk` | `B*min(T,512)*V*d` | 254,279,680 | `tools/ft_lowmem_loss.py:10,13-28,40-47` |
| `loss_logits_gradient` | `B*min(T,512)*V*d` | 254,279,680 | `tools/ft_lowmem_loss.py:24-28,40-47`; `mlx/primitives.cpp:3596-3619` |
| `loss_head_input_gradient` | `B*min(T,512)*H*d` | 2,097,152 | `tools/ft_lowmem_loss.py:24-28,40-47`; `mlx/primitives.cpp:3596-3619` |
| `recompute_moe_forward_activations` | `R*(2H+3I)*d + B*T*(3S+2H+1)*d + B*T*(E+k)*d` | 101,906,000 | `mlx_lm/models/qwen3_next.py:299-346`; `mlx_lm/models/switch_layers.py:182-220` |
| `moe_score_weighted_routed_output_and_reduction` | `(R*H+B*T*H)*d` | 36,864,000 | `mlx_lm/models/qwen3_next.py:335-341`; `mlx_lm/models/switch_layers.py:198-220` |
| `retained_topk_indices` | `R*4` int32 bytes | 32,000 | `mlx_lm/models/qwen3_next.py:328-335`; `mlx_lm/models/switch_layers.py:198-208` |
| `recompute_gdn_projection_activations` | source-shaped GDN projection/normalization envelope, `B*T*G*d + 2*B*T*H_v*4` | 110,976,000 | `mlx_lm/models/qwen3_5.py:88-201`; `mlx_lm/models/gated_delta.py:605-647` |
| `recompute_attention_projection_activations` | `B*T*(3H+4Q*D+4K_v*D)` | 49,152,000 | `mlx_lm/models/qwen3_next.py:74-151` |
| `recompute_lora_rank_activations` | `B*T*r*max adapted branch module count*4` | 288,000 | `mlx_lm/tuner/lora.py:86-98`; target list `docs/finetune/lora.yaml:17-30` |
| `lora_matmul_vjp_tensors` | `4*B*T*max_branch(sum_j(input_j+r))` | 76,064,000 | `mlx_lm/tuner/lora.py:95-98`; `mlx/primitives.cpp:3086-3111` |
| `quantized_matmul_input_gradients` | max GDN `B*T*d*(8H+V_g)` and attention `B*T*d*(7H+Q*D)` | 40,960,000 | `mlx/primitives.cpp:3596-3619`; `mlx_lm/models/qwen3_5.py:145-201`; `mlx_lm/models/qwen3_next.py:122-151` |
| `quantized_moe_shared_input_gradients` | `B*T*(3H+S)*d` for shared gate/up/down and scalar gate input cotangents | 13,312,000 | `mlx_lm/models/qwen3_next.py:313-315,338-341`; `mlx/primitives.cpp:3596-3619` |
| `gather_qmm_vjp_input_gradients` | `R*(2H+I)*d` for routed gate, up, and down input gradients | 73,728,000 | `mlx_lm/models/switch_layers.py:93-109,182-220`; `mlx/primitives.cpp:3782-3840`; `mlx/ops.cpp:5794-5826` |
| `switch_gather_scatter_vjp` | `(R*H+B*T*H)*d` | 36,864,000 | `mlx_lm/models/switch_layers.py:11-23,198-220`; `mlx/primitives.cpp:2564-2581` |
| `switch_index_sort_arrays` | `3R*4` for sorted route IDs, sorting order, and inverse order; no expert-offset array is created in this path | 96,000 | `mlx_lm/models/switch_layers.py:11-16,201-218` |
| `attention_sdpa_lse` | `B*Q*T*4` | 64,000 | `mlx_lm/models/qwen3_next.py:120-151`; `mlx/backend/metal/scaled_dot_product_attention.cpp:1663-1670` |
| `attention_sdpa_vjp_score_tiles` | `3*B*Q*min(T,1024)^2*d` | 96,000,000 | `mlx/backend/metal/scaled_dot_product_attention.cpp:1119-1135,1177-1180` |
| `attention_sdpa_vjp_tile_gradients` | `B*Q*min(T,1024)*(2D+D)*d` | 24,576,000 | `mlx/backend/metal/scaled_dot_product_attention.cpp:1182-1185` |
| `attention_sdpa_vjp_float_accumulators` | `(B*Q*T*D+2*B*K_v*T*D)*4` | 20,480,000 | `mlx/backend/metal/scaled_dot_product_attention.cpp:1187-1190` |
| `attention_sdpa_vjp_output_arrays` | `(B*Q*T*D+2*B*K_v*T*D)*d` | 10,240,000 | `mlx/backend/metal/scaled_dot_product_attention.cpp:1671-1688` |
| `attention_sdpa_vjp_odo` | `B*Q*T*4` | 64,000 | `mlx/backend/metal/scaled_dot_product_attention.cpp:1682-1708`; `kernels/scaled_dot_product_attention_vjp.metal:8-35` |
| `router_softmax_vjp_arrays` | `(3*B*T*E+B*T)*d` | 1,538,000 | `mlx_lm/models/qwen3_next.py:325-341`; `mlx/primitives.cpp:5529-5541` |
| `router_quantized_matmul_input_gradient` | `B*T*H*d` | 4,096,000 | `mlx_lm/models/qwen3_next.py:310,325-326`; `mlx/primitives.cpp:3596-3619` |

### Frozen prefix, replay phases, and loss

`lora.py` freezes the model before converting only the requested final layers (`mlx_lm/lora.py:233-252`). `mlx.nn.value_and_grad` differentiates only `model.trainable_parameters()` (`mlx_py/nn/utils.py:12-38`), and the trainer checkpoint wrapper passes each layer's trainable parameter tree to `mx.checkpoint` (`mlx_lm/tuner/trainer.py:27-40`). From those sources, the frozen prefix before the first LoRA layer has no parameter cotangents and the loss does not request a gradient to the model input. The budget therefore counts no internal saved prefix activations; it counts the single `[B,T,H]` hidden boundary entering the adapted segment. That is an inference from the differentiated argument tree, not a buffer capture.

For each replayed decoder layer, `peak_bytes()` sums the GDN/attention scratch inventory and separately sums MoE/router scratch, then uses their maximum. This prevents adding the attention recurrence/SDPA arrays to expert routing arrays merely because both occur in the same layer. It also takes the outer maximum over the adapted layers, while retaining all `L+1` hidden checkpoint boundaries. The returned term rows remain an inventory: e.g. GDN projection and routed-expert arrays are both listed, but do not both contribute to the GDN phase peak.

The custom completion loss slices hidden states in chunks of at most 512 positions and checkpoints the vocabulary projection (`tools/ft_lowmem_loss.py:10,24-47`). Those slice VJPs accumulate into one full `[B,T,H]` hidden cotangent; the separate `loss_head_input_gradient` row is one chunk's input cotangent. Qwen3.5 applies the final norm after decoder layers (`mlx_lm/models/qwen3_5.py:309-315`), so `final_norm_output` is another full `[B,T,H]` array outside the decoder-layer checkpoint boundaries.

### GDN recurrence and the checkpoint contradiction

Training selects `use_kernel=False` (`mlx_lm/models/qwen3_5.py:183-194`), which dispatches to the ops recurrence (`mlx_lm/models/gated_delta.py:639-647`). A recurrent state is `[B,H_v,D_v,D_k]` FP32 (`:397-405`), or `M=2,097,152 B=2 MiB` for this model. The ops path iterates each token and carries state to the next step (`:557-602`). In one step, the code computes the state decay product `state*decay` (`:408-416`), the state/key product before its reduction (`:417`), the key/delta outer product added to state (`:418-419`), and the state/query product before output reduction (`:420-421`). The four-array term is a measured envelope, not a proof that only those four arrays exist. `Multiply::vjp` emits multiply results at operand shapes; `Reduce::vjp` broadcasts cotangents to the input shape (`mlx/primitives.cpp:3248-3258,4109-4118`). If a mask is present, `Select::vjp` creates separate true/false cotangent products (`:3301-3328`); training's `create_ssm_mask` returns `None` without a cache (`mlx_lm/models/base.py:58-61`).

The implemented default formula is `gdn_step_vjp_state_arrays = 4*T*M = 8,388,608*T B = 8 MiB/token`, plus the separate vector, projection, and GDN chunk-boundary terms. At `T=1000`, the state-array term is 8,388,608,000 B (7.8125 GiB). The optional `--assume-working-gdn-checkpoint` instead uses `4*min(T,64)*M` for this row and retains `(ceil(T/64)+1)*M` boundaries; this is the proposed working-checkpoint model, not the default prediction.

`tools/gdn_checkpoint.py` wraps each 64-position chunk, passes the chunk slices and entry state, and returns the chunk output plus exit state (`:20-50,81-95`). MLX custom functions stop-gradient their forward outputs and append those outputs to the primitive inputs (`mlx/transforms.cpp:1033-1045`); `checkpoint`'s VJP calls `vjp(fun, depends(primals, outputs), cotangents)` (`:1119-1129`). `depends()` appends the forward outputs to its inputs (`mlx/ops.cpp:6604-6627`), and its VJP sends cotangents to the original primals while returning zeros for dependency-only outputs (`mlx/primitives.cpp:1826-1840`). The generic `vjp` differentiates all primals unless `argnums` restricts them (`mlx/transforms.cpp:574-580`); it traces a tape, walks it backward, and accumulates each primitive's input cotangents (`:447-557`). Since the chunk inputs include `q,k,v,g,beta` as well as state, training requests gradients for full-sequence projected inputs; each chunk's recurrence also contains the per-token state products listed above. The evaluator builds a dependency-counted schedule and detaches evaluated arrays after their last scheduled use (`:116-229,235-316`). These sources show the chain and the per-step VJP shapes, but do not prove the exact overlap of chunk-input cotangents, recurrence products, and Metal buffers.

This is **open question #1** because the measurements contradict the working-checkpoint formula. Isolated real-weight GDN block peaks above loaded weights with checkpoint ON are 0.887/1.617/3.681 GiB at `T=128/256/512`, roughly 7.2 MiB/token; checkpoint OFF is 1.141/2.192/4.292 GiB, roughly 8.2 MiB/token by the measured slope. Trainer sweep deltas for the GDN-containing adapter set are 1.62/3.47/6.97 GiB at `T=256/512/1000`, or 6.3/6.9/7.1 MiB/token, and stay flat from `L=2` through `L=16`. The default `4T` source envelope predicts these trainer points within the regression range; the `4*min(T,64)+boundaries` option predicts only about 0.532/0.562/0.621 GiB of isolated incremental memory at those three block lengths (−40.1%/−65.3%/−83.1% versus measured). Checkpointing therefore does not bound the measured backward term as the simple chunk-boundary model suggests. The source-visible per-chunk input cotangents and per-step broadcast/multiply VJPs are candidates; the specific retained buffers that produce the measured coefficient remain untraced. No coefficient is fitted to these measurements.

### Frozen quantized gather-QMM gradients and weights

`GatherQMM::vjp` recomputes `gather_qmm` for the input gradient using the original packed weights, scales, and expert indices; it explicitly rejects quantized-weight gradients (`mlx/primitives.cpp:3782-3855`). The operation output shape includes the routed index dimensions, input row dimension, and expert output width (`mlx/ops.cpp:5794-5826`); the Metal path dispatches the gather kernel with the packed weights and scales (`mlx/backend/metal/quantized.cpp:1980-2053,1367-1388`). It does not allocate a dense dequantization of every expert. For this row `R=B*T*k=8000`: the routed input-gradient arrays are `R*(2H+I)*d=73,728,000 B`, sized per selected token/top-k route, not per all `E=256` experts. The router's original top-k indices are separately retained as `R*4` bytes; sorting adds sorted route IDs, route order, and inverse order when `indices.size>=64` (`mlx_lm/models/qwen3_next.py:328-335`; `mlx_lm/models/switch_layers.py:11-16,201-218`). This path does not construct an expert-offset array, so the budget does not include one.

Safetensors contain 20,401,929,952 B (19.0008 GiB). The Qwen3.5-MoE sanitizer discards `vision_tower` and `model.visual` keys (`mlx_lm/models/qwen3_5_moe.py:23-34`); this directory has 893,142,496 such bytes. The base sanitizer drops MTP keys and a tied `lm_head`, but this config is untied and the archive has no MTP tensors (`mlx_lm/models/qwen3_5.py:350-357`). Thus resident payload is 19,508,787,456 B (18.1690 GiB), matching measured MLX active after load, 19,508,787,464 B, within 8 B. The former 19.00 GiB estimate counted vision tensors that the loader sanitizes away.

### Allocator cache, system wired, and preflight

The Metal allocator reads `memory_size` and `max_recommended_working_set_size`, then sets the default pool cap to `min(1.5*recommended,0.95*device_memory)` and GC threshold to `min(0.95*recommended,cap)` (`mlx/backend/metal/allocator.cpp:58-65`). `mx.set_cache_limit` changes the pool cap (`:81-85`), while allocation first tries cached buffers, considers `active+cache+new_size` against the GC threshold, and retains freed buffers under the cap (`:128-175,183-200`). The code reads `mx.device_info()` and `mx.get_cache_memory()` at runtime when they can be queried; this headless run could not initialize Metal, so the table uses the existing 26.8/32 GiB fallback values. The runtime device query is not verified in this environment.

For `T=1000,L=2`, active peak is 26.287 GiB at backward replay of layer 38, where GDN/attention scratch is 8.094 GiB and MoE/router scratch is 0.275 GiB; the phase max is used. With the fallback default cache cap, wired prediction is 27.987 GiB at the current empty cache occupancy and 58.387 GiB at the cache-cap upper bound, plus the 1.7 GiB OS reserve. Those are current-occupancy and capacity-bound values, not an assertion that the cap is occupied. At cache limit 0, the bound collapses to active plus OS reserve. `mlx_lm.tuner.trainer` clears any positive cache after `mx.eval` because `clear_cache_threshold` defaults to 0 (`trainer.py:22-24,80-85,331-332`); this models the cache at the end of each step, while the capacity bound covers buffers retained during a step before the clear.

`tools/mem_preflight.py` tokenizes every row from `train.jsonl` and `valid.jsonl` with the model chat template and tokenizer files, without loading model weights. It applies trainer padding and the max-sequence cap, predicts active plus cache-cap bound plus the 1.7 GiB reserve, and compares against `29.5-1.0=28.5 GiB`. It exits 3 and lists over-budget rows plus the largest fitting row length if the bound reaches that threshold. `tools/ft_lora_lowmem.py` runs this gate before `lora.main()`, using CLI values ahead of the training YAML; `MEM_PREFLIGHT=0` disables the gate. With the local data, `L=2`, `max_seq_length=1024`, and `--cache-limit-gb 0`, the check tokenized 41 rows; all fit, the longest row was 1024 IDs (`model T=1023`), and the wired bound was 28.172 GiB. With the default 30.4 GiB cap, the cap-bound preflight intentionally reports no fitting sequence; set `MLX_CACHE_LIMIT_GB=0` or pass an explicit lower cap when that is the run configuration.

### Prediction versus measurement

Trainer regression uses measured trainer-step, cache-limit-0 active peaks from `results/20261105T143800Z-memory-model/NOTES.md:40-50`. Predictions pad row IDs as the trainer does. The signed error is `(predicted-measured)/measured`.

| Row IDs T | L | Model T | Predicted active GiB | Measured active GiB | Error |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 256 | 18.413 | 18.60 | −1.0% |
| 512 | 1 | 512 | 18.655 | 18.87 | −1.1% |
| 1000 | 1 | 1024 | 18.663 | 19.39 | −3.7% |
| 2000 | 1 | 2016 | 18.750 | 19.85 | −5.5% |
| 3000 | 1 | 3008 | 19.034 | 20.11 | −5.3% |
| 256 | 2 | 256 | 20.254 | 20.22 | +0.2% |
| 512 | 2 | 512 | 22.330 | 22.34 | −0.0% |
| 1000 | 2 | 1024 | 26.480 | 26.36 | +0.5% |
| 512 | 4 | 512 | 22.342 | 22.32 | +0.1% |
| 1000 | 4 | 1024 | 26.497 | 26.45 | +0.2% |
| 768 | 3 | 768 | 24.412 | 24.46 | −0.2% |
| 900 | 8 | 928 | 25.750 | 25.66 | +0.4% |
| 900 | 16 | 928 | 25.812 | 25.70 | +0.4% |
| 768 | 16 | 768 | 24.505 | 24.40 | +0.4% |

The new regression test checks all fourteen predictions against the 10% limit. The earlier held-out check remains a guard check rather than a fit: `L=3,T=768` predicts 24.412 GiB active versus 24.46 measured (−0.2%); `L=2,T=1500` predicts 30.373 GiB active and therefore a guard stop, as measured (the measured value is censored at 29.52 GiB wired). These predictions use the source-derived phases and the default `4T` GDN envelope.

| Isolated block, incremental above weights | T | Predicted GiB | Measured GiB | Error |
|---|---:|---:|---:|---:|
| GDN, checkpoint off | 128 | 1.026 | 1.141 | −10.1% |
| GDN, checkpoint off | 256 | 2.052 | 2.192 | −6.4% |
| GDN, checkpoint off | 512 | 4.104 | 4.292 | −4.4% |
| GDN, checkpoint off | 1000 | 8.015 | >11.000 stop | <−27.1% vs lower bound |
| GDN, checkpoint on | 128 | 1.032 | 0.887 | +16.3% |
| GDN, checkpoint on | 256 | 2.062 | 1.617 | +27.5% |
| GDN, checkpoint on | 512 | 4.121 | 3.681 | +12.0% |
| MoE | 1000 | 0.250 | 0.190 | +31.6% |
| MoE | 2000 | 0.500 | 0.630 | −20.6% |
| Attention | 1000 | 0.221 | 0.300 | −26.3% |
| Attention | 2000 | 0.335 | 0.480 | −30.2% |

The isolated-versus-trainer comparison at row length 1000 is also retained. Current predictions are 18.663 GiB for isolated `L=1,T=999`, 18.663 GiB for trainer `L=1,T=1024`, 26.279 GiB for isolated `L=2,T=999`, and 26.480 GiB for trainer `L=2,T=1024`. Measurements are 19.293, 19.274, 23.210, and 26.357 GiB respectively. The `L=2` cache-0 trainer error is +0.5%; isolated `L=2` is +13.2%. Default cache and 2 GiB cache runs stopped at 29.59 and 29.88 GiB wired. Their model outputs are capacity bounds (58.58 and 30.18 GiB at `L=2,T=1024`), not point predictions of cache occupancy.

The prior Brief 36 rows remain in `--points` and are summarized here. They use row-length ranges as recorded, not trainer padding.

| Point | Predicted active GiB | Measured GiB | Error / metric |
|---|---:|---:|---|
| probeA, L1, row 470 | 18.615 | 19.480 | −4.4% active |
| probe4, L4, row 470–471 | 21.995–22.003 | 25.140 | −12.5% active |
| train2, L2, row 600 | 23.036 | 24.460 | −5.8% active |
| probeI, L2, row 600 | 23.036 | 22.532 | +2.2% active |
| probeJ, L2, row 600 | 23.036 | 22.528 | +2.3% active |
| probeC, L1, row 1200–3000 | 18.666–19.032 | 24.190 | −22.8% to −21.3% active |
| run3, L1, row 1200–3000 | 18.666–19.032 | 24.180 | −22.8% to −21.3% active |
| probeE, L1, row 1200–3000 | 18.666–19.032 | 24.177 | −22.8% to −21.3% active |
| train1, L4, row 600 | 23.050 | OOM at 28 GiB limit | censored |
| probeB / probeG, L2, row 1200–3000 | 27.900–42.492 active | 25.611 | +8.9% to +65.9%; reported decimal-GB footprint, not MLX peak |
| probeH, L2, row 1200–3000 | 29.600–44.192 occupancy floor; 60.000–74.592 cap bound | 29.930 | wired; no point gap for a bound |
| probeF, L2, row 1200–3000 | 27.900–42.492 active | 31.260 | process resident, not MLX active |

### Open questions

1. **GDN checkpoint remains the leading contradiction.** The default measured model is `4T*M`; the measured checkpoint-on isolated block slope is about 7.2 MiB/token and the trainer GDN deltas are 6.3/6.9/7.1 MiB/token. The optional working formula is far below the isolated checkpoint-on measurements. Source shows chunk custom-VJP inputs, the recurrent state chain, per-step multiply/reduction cotangents, and the evaluator schedule, but not the live Metal buffer set. Candidate terms are per-chunk projected-input cotangents and the decay/state-key/key-delta/query products; their precise overlap is not established.
2. **L1 trainer regression at T=2000 and 3000: −5.5% and −5.3%.** The prediction is set by the attention replay phase and full-sequence terms; candidate undercount terms are attention projection activations, quantized input gradients, and SDPA VJP accumulators. Source does not identify which arrays account for the remaining growth.
3. **probe4: −12.5%.** The source budget takes one layer replay at a time. The measured active peak is above it; overlap among replay outputs or evaluator-retained arrays across layer calls has not been traced.
4. **train2: −5.8%.** The measured `L=2,T≈600` point is above the model. Candidate terms are GDN recurrence products and attention/MoE replay scratch; source does not establish their simultaneous live count.
5. **probeC, run3, probeE: −22.8% to −21.3%.** The L1 long-row active measurements exceed the model. Attention projection/VJP arrays and chunked-loss full-sequence cotangents are candidate terms; no source trace establishes the missing allocation.
6. **probeB and probeG: +8.9% to +65.9% numerical difference.** The recorded measurement is a reported decimal-GB footprint, not the MLX active peak modeled here. The comparison is not like-for-like; the footprint accounting term remains untraced.
7. **GDN checkpoint-off block T128: −10.1%; T256: −6.4%.** The four-state-array envelope is below these points. Additional recurrence VJP arrays or evaluator-held buffers are candidates, but their exact overlap is unverified.
8. **GDN checkpoint-on blocks T128/T256/T512: +16.3%/+27.5%/+12.0%.** The default `4T` envelope plus chunk boundaries exceeds the isolated results. The `4*min(T,64)+boundaries` counterfactual is much smaller than the measurement; neither source reading explains the observed intermediate coefficient.
9. **GDN checkpoint-off T1000: prediction 8.015 GiB versus a `>11 GiB` guard stop.** The predicted term is at least 27.1% below the measured lower bound. Per-step state-shaped multiply/reduction VJPs and evaluator-held arrays are candidates; the exact live overlap is unknown.
10. **MoE blocks T1000/T2000: +31.6%/−20.6%.** The source-shaped routed outputs, indices, `gather_qmm`, scatter, and shared-expert gradients do not predict both lengths within 5%. Kernel workspace, routing-array overlap, or the measurement's allocator peak may contribute; no cause is traced.
11. **Attention blocks T1000/T2000: −26.3%/−30.2%.** Projection/VJP inventories are below measured increments. Candidate terms are the SDPA tile workspaces, projection/gate activations, and quantized projection input gradients; their exact simultaneous allocation is not established.
12. **Isolated L2 step: +13.2% (26.279 predicted, 23.210 measured).** The model's GDN replay phase exceeds the isolated peak. Nested GDN and outer layer checkpoint liveness is a candidate; source and measurements do not identify the released or overlapping arrays.

The cache-stop rows and `probeH` are censored or capacity-bound comparisons, so they do not have pointwise percentage gaps. The new fourteen-point trainer regression meets the requested 10% limit; the remaining numeric gaps above 5% are listed individually.

### Unverified details

- The GDN `4T` coefficient is a source-shaped envelope supported by the sweep slope, not a count of exactly four simultaneously allocated Metal buffers. The query product, VJP cotangents, and command-buffer temporaries may overlap differently.
- Three Adam-sized temporary results are counted. MLX lazy evaluation and buffer reuse make their exact simultaneous liveness uncertain.
- Cache occupancy changes with reuse and evaluation order. The preflight uses the allocator cap as a conservative per-step bound and reports current occupancy separately; it does not claim the cap is the measured occupancy.
- Runtime `mx.device_info()` values were not verified in this headless run; fallback device constants were used. On a Metal-capable run, the tool reads runtime device limits and current cache occupancy.

## 2. Measurements (lead, on the Mac, 2026-11-05)

All runs under `tools/mem_guard.py` (stop at 29.5 GiB system wired), `iogpu.wired_limit_mb=31744`, normal apps
open. Harness: `tools/mem_profile.py` (`step`, `trainer-step`, `block`, `sweep`, `fit`). Raw numbers:
`results/20261105T143800Z-memory-model/` (NOTES.md, metrics.json). "Active" = `mx.get_peak_memory()`;
"wired" = system wired pages (what the OS and mem_guard see).

- **Blocks in isolation** (real weights, backward w.r.t. the input, above the 18.17 GiB of weights): MoE
  0.19 / 0.63 GiB and attention 0.30 / 0.48 GiB at T = 1000 / 2000; GDN 1.14 / 2.19 / 4.29 GiB at T = 128 / 256 /
  512 (~8.2 MiB/token), > 11 GiB at T = 1000. GDN is the only term that matters.
- **Trainer vs isolated step**: identical at L = 1; +3.15 GiB at L = 2, T = 1000 (open, §4).
- **Allocator cache**: with MLX's default cache limit, freed buffers stay wired (6.55 GiB in one step) and the
  trainer hits the guard at L = 2, T = 1000. With `MLX_CACHE_LIMIT_GB=0` it fits (26.36 GiB active).
  `mx.get_peak_memory` never shows the cache; the OS does.
- **Sweep** (trainer, cache 0; active / wired GiB; "stop" = mem_guard):

| T | L=1 | L=2 | L=3 | L=4 |
|---|---|---|---|---|
| 256 | 18.60 / 21.32 | 20.22 / 23.20 | 20.25 / 23.99 | 20.25 / 23.84 |
| 512 | 18.87 / 21.29 | 22.34 / 24.59 | 22.39 / 25.51 | 22.32 / 24.51 |
| 1000 | 19.39 / 21.95 | 26.36 / 28.92 | stop | 26.45 / 29.03 |
| 2000 | 19.85 / 22.04 | stop | stop | stop |
| 3000 | 20.11 / 22.52 | stop | stop | stop |

  Deep LoRA: L = 8, T = 900: 25.66 / 28.58; L = 16, T = 900: 25.70 / 29.27; L = 16, T = 768: 24.40 / 28.36.
- **Not done as asked (stated plainly):**
  - *Per-layer waterfall:* `mem_profile.py waterfall` hits mem_guard even at L = 2, T = 512, where a normal step
    needs 22.3 GiB. The brief (36b) asked for gradient cuts below every layer down to layer 0. That runs the
    backward through ~30 frozen GDN layers, which real training never does, so these numbers would not describe
    training. Fix: cut only within the adapted layers. Not yet re-run.
  - *GPU capture:* a full-model capture would carry ~18 GB of weight buffers, so it was taken on the Open #1
    toy repro instead (`repro/toy2-ckpt-T128.gputrace`, 288 MB, not committed; T = 128, checkpointed, gradients
    w.r.t. q, k, v: 605 MiB). It still has to be read in Xcode (Memory / Resources view) for state-sized buffers
    that stay live across chunks. Not analyzed yet.
- **Entry point checked end to end:** `ft_lora_lowmem.py` on night1, L = 2, 2 iterations, with
  `MLX_CACHE_LIMIT_GB=0`: preflight passes (28 rows, bound 27.1 GiB), trains at 10.4 s/step, peak 22.29 GiB.
  Without the env var it refuses with exit 3 and prints the cache hint.

## 3. The law

The requested form `peak = W + a·L + b·T + c·L·T` fits badly (R² 0.72) because peak does not grow with L: with
`grad_checkpoint` one layer's backward is live at a time, so the peak is a **max over adapted layers**, not a sum.
Measured law (cache limit 0):

    active ≈ 18.45 GiB + 0.55 MiB·T + [any GDN layer adapted] · ~7 MiB·T
    wired  ≈ active + 2.4…4.0 GiB (grows with L)

The ~7 MiB/token GDN term (≈ 4 recurrent states of 2 MiB per token) is what the budget models (§1). L = 1 adapts
only layer 39 (full attention), which is why it fits at 3k tokens. Every L ≥ 2 includes a GDN layer.

**What fits at the guard** (cache limit 0, 1 GiB margin, `tools/mem_preflight.py`):

| L | longest row that fits | rows of the current set (41) that fit |
|---|---|---|
| 1 | ≥ 9k tokens (measured to 3k, extrapolated beyond) | 41 |
| 2-8 | 768 tokens | 28 |
| 16 | 736 tokens | 28 |

## 4. Proof and open questions

- **Held-out predictions** written before the runs: L = 3, T = 768: lead's law 24.07, budget 24.59, measured
  24.46 GiB (−1.6% / +0.5%). L = 2, T = 1500: both predicted a guard stop; measured stop at 29.52 GiB wired.
- **Independent review** (Codex C20, read-only, no access to notes or results): term formulas largely
  confirmed. It found the per-phase over-summing and missing terms that §1 now includes. It also said that, by the
  code, `gdn_checkpoint.py` should cut GDN to 4·min(T,64) states. That contradicts every measurement (next point).
- **Open #1: `mx.checkpoint` does not bound a recurrence's backward when its per-step inputs need gradients.**
  Minimal repro (`results/20261105T143800Z-memory-model/repro/ckpt_toy2.py`, 2 MiB state): gradient w.r.t. the
  initial state only: checkpointed backward is flat (0.37-0.43 GiB, T = 256-1024). Gradients w.r.t. per-step
  q, k, v (what training needs): 1.03 / 1.58 / 2.69 GiB, growing with T. Mechanism in MLX 0.32.4 not yet traced.
  This is why `GDN_CHECKPOINT` saves only 0-15% on the backward of an *adapted* GDN layer.
- **Correction (17:20): keep `GDN_CHECKPOINT=1` (default on).** It does matter for the *frozen* GDN layers.
  In training mode every GDN layer runs the ops path (`use_kernel=not self.training`), including the 30 frozen
  ones that need no gradient. The checkpointed version roughly halves their forward cost. Real 1-layer
  training, one step on the v1 rows: row 29 (5,269 tokens) and row 3 (4,380 tokens) hit mem_guard with the
  checkpoint off (> 29.5 GiB wired), and peak at 20.34 / 20.26 GiB with it on. Flipping the default off (16:30)
  caused the first two config-A stops. All sweep numbers above were taken with it on, so the law holds with it on.
  Follow-up: frozen GDN layers could run the inference kernel (no autodiff graph) instead.
- Open #2: wired − active grows with L (2.4 → 4.0 GiB), outside MLX's counters. The budget uses the largest
  measured gap (4.0 GiB) as the OS reserve, which makes the preflight conservative.
- Open #3: the trainer is 2.94 GiB above the isolated step at L = 2, T = 1000 after padding (untraced).
- Open #4: L = 1 on long rows is underpredicted by ~5%. Historical 1-layer runs on 1.2-3k rows (probeC/run3/
  probeE, 24.2 GiB with the default cache) are ~21% above the cache-0 model; their default cache is the likely
  but unproven reason.

## 5. Recommendation for the ~250-row run

Run with `MLX_CACHE_LIMIT_GB=0` and the preflight on (default in `tools/ft_lora_lowmem.py`). Two configs fit today:

- **A. L = 1, all rows up to the 10,240-token cap.** Fits with ≥ 7 GiB to spare, uses every row. It adapts only
  the last layer (attention + shared expert).
- **B. L = 8 (or 16), rows ≤ 768 tokens only.** Same memory as L = 2, about 2x the step time of L = 4. Drops the
  ~1/3 of rows that are longer (13 of the current 41).

Recommendation: **A first**. It uses all the data, and the earlier 1-layer run (run3) showed no repetition
collapse. Run B on the same seed set as the comparison. Both are judged on the frozen heldout-v2 set against the
base model, with the paired 95% CI from `tools/ft_eval.py`. Neither result is in yet.

To train many layers on long rows, GDN needs a backward that `mx.checkpoint` cannot give us today (Open #1). The
options are a hand-written chunked vjp (`mx.custom_function`, reverse recurrence per chunk) or a chunked
forward/backward Metal kernel like flash-linear-attention's. Either would cut ~7 GiB to ~0.5 GiB at T = 1000.
That is a separate decision.
