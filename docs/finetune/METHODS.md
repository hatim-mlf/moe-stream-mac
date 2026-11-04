# Fine-tuning methods for Qwen3.6-35B-A3B on MLX

## Recommendation in brief

Use **masked SFT with rank-8 LoRA on the last four transformer blocks**, covering their attention/linear-attention projections and shared expert. Keep the routed experts and router frozen. Begin with the repository's current `scale: 20` only as the inherited experimental setting; in MLX this is a direct multiplier, not a LoRA `alpha` that is automatically divided by rank. The present config therefore behaves like α=160 under the common α/r convention. Compare against `scale: 2` (α=16 at rank 8) before treating the larger value as a good choice.

Do not run preference training yet: the local log sidecar has only three genuine pairs, and installed `mlx-lm==0.32.2` contains no DPO/IPO/ORPO/KTO/GRPO trainer. If a larger train-only set of passing/failing pairs is collected and an MLX trainer is separately tested, ORPO is the first objective to evaluate because it does not need a reference model. It is not a command available in this environment today.

## SFT: completion-only token cross-entropy

For prompt (x=(x_1,\dots,x_m)), target answer (y=(y_1,\dots,y_n)), and autoregressive policy πθ, the usual teacher-forced next-token loss on answer tokens is

\[
\mathcal L_{\mathrm{SFT}}(\theta)
=-\frac{1}{\sum_{t=1}^{n} w_t}
\sum_{t=1}^{n} w_t\log \pi_\theta(y_t\mid x,y_{<t}),
\]

where (w_t=1) on answer tokens and 0 elsewhere. The standard instruction-tuning likelihood objective is described in [InstructGPT](https://arxiv.org/abs/2203.02155). Prompt masking sets the prompt's (w_t) to zero; it does **not** remove the prompt from the model input or stop the answer from being conditioned on it.

That distinction matters here. The system rules, full code brief, prior implementation, and review/test output can be much longer than the requested code. Training loss on those supplied tokens wastes updates teaching the model to reproduce context it will receive at inference, and can reward copying instructions instead of applying them. We want the model to learn which implementation follows a precise contract and how it changes after feedback. The installed `ChatDataset` applies the model's chat template, computes an offset through the preceding messages, and `default_loss` counts only unmasked target positions. It masks the prompt and scores the final assistant message when `mask_prompt: true`.

**Installed support:** `mlx-lm==0.32.2`, `mlx==0.32.4`; `mlx_lm.lora` supports local `messages` JSONL, prompt masking, gradient checkpointing, LoRA/DoRA/full fine-tuning, batch size, sequence length, and Adam/AdamW/etc. The current local data uses this `messages` schema. The trainer's default loss is masked cross-entropy only; there is no MoE load-balancing loss or preference loss.

## LoRA, DoRA, and QLoRA

### LoRA update and initialization

For a frozen matrix (W\in\mathbb R^{d_{out}\times d_{in}}), rank-(r) LoRA learns (A\in\mathbb R^{r\times d_{in}}) and (B\in\mathbb R^{d_{out}\times r}):

\[
W'=W+\frac{\alpha}{r}BA,
\qquad \Delta W=\frac{\alpha}{r}BA.
\]

This replaces (d_{out}d_{in}) trainable weights with (r(d_{in}+d_{out})). [Hu et al. (LoRA)](https://arxiv.org/abs/2106.09685) initialize one factor randomly and the other to zero so the initial adapter is a no-op. Installed MLX does the same in its stored layout: `lora_a` random, `lora_b` zero. It uses `scale` as the **direct** coefficient on (xAB); it does not divide that value by rank. Thus translate a conventional α/r value into MLX `scale`. Rank 8 is a sensible first probe; rank 16 is an ablation only after rank 8 clears the memory and held-out gates. Larger ranks add little cost to the recommended last-four-layer adapter but do not fix weak/incorrect labels.

### DoRA and quantized-base LoRA

DoRA separates a weight row's magnitude from its direction. For output row (j), a simplified expression is

\[
W'_j=m_j\frac{W_j+\Delta W_j}{\lVert W_j+\Delta W_j\rVert_2},
\qquad \Delta W=BA,
\]

with trainable magnitudes (m_j). See [DoRA](https://arxiv.org/abs/2402.09353). It can improve adaptation in some settings, but adds norm work and an extra magnitude vector; it is not the first choice for this small pilot. In the installed MLX implementation, DoRA supports dense `Linear` and quantized `Linear`; conversion of `SwitchLinear`/routed-expert modules explicitly raises an error.

QLoRA trains low-rank updates over a frozen low-bit base rather than updating all full-precision base weights; the reference method uses NF4, double quantization, and paged optimizers ([QLoRA](https://arxiv.org/abs/2305.14314)). This model is **4-bit affine**, not NF4, so call this QLoRA-style MLX adaptation rather than an exact reproduction of QLoRA's quantizer. MLX-LM converts quantized linear layers to a LoRA wrapper while keeping the quantized base frozen.

### Parameter and adapter-memory counts for this exact config

Counts use `/Users/hatim/models/Qwen3.6-35B-A3B-4bit/config.json` and the installed Qwen3.5 MoE implementation: hidden size 2,048; 40 blocks; 30 Gated DeltaNet blocks and 10 full-attention blocks (three linear-attention then one full-attention); linear-attention key/value widths 2,048/4,096; full-attention head width 256 with 16 query heads and 2 KV heads; 256 routed experts, top-8; expert and shared-expert hidden width 512. For each ordinary projection, adapter parameters are (r(d_{in}+d_{out})). Routed `SwitchLinear` has that factor for each of 256 experts.

| Adapted family | Rank-8 params in one block | Last four blocks | All 40 blocks | Adapter bytes at last four, BF16 / FP32 |
|---|---:|---:|---:|---:|
| Gated DeltaNet: `in_proj_qkv` (2,048→8,192), `in_proj_z` (2,048→4,096), `in_proj_b/a` (2,048→32 each), `out_proj` (4,096→2,048) | 213,504 | 640,512 across the three linear blocks | 6,405,120 across 30 linear blocks | 1.22 / 2.44 MiB |
| Full attention: `q_proj` (2,048→8,192 including its gate), `k_proj`/`v_proj` (2,048→512 each), `o_proj` (4,096→2,048) | 172,032 | 172,032 for the one full-attention block | 1,720,320 across 10 blocks | 0.33 / 0.66 MiB |
| Shared expert: `gate_proj`, `up_proj` (2,048→512), `down_proj` (512→2,048), `shared_expert_gate` (2,048→1) | 77,832 | 311,328 across four blocks | 3,113,280 across 40 blocks | 0.59 / 1.19 MiB |
| **Recommended attention + shared expert total** | — | **1,123,872** | **11,238,720** | **2.14 / 4.29 MiB** |
| Router `mlp.gate` (2,048→256) | 18,432 | 73,728 | 737,280 | 0.14 / 0.28 MiB |
| All three routed-expert projections, `switch_mlp.{gate,up,down}_proj`, for 256 experts | 15,728,640 | **62,914,560** across four blocks | **629,145,600** across 40 blocks | **120 / 240 MiB** |

The byte columns count adapter tensors only, not optimizer state, gradients, activations, or the frozen base. Adapter arrays are initialized as MLX floating tensors; a conservative FP32 training-state budget of parameters + gradients + two Adam moments is roughly 16 bytes per parameter: about 17 MiB for the recommended rank-8 adapter, 0.94 GiB for routed experts in the last four blocks, and 9.38 GiB for all routed experts in all blocks. Optimizer implementation/dtype can change these figures. Activations and the base model dominate the recommended run.

The local model directory is **19 GB on disk**. That already exceeds the 12–16 GB realistic available-memory estimate while other applications are open; actual MLX peak, weight loading, activation memory, and swap behavior were not measured in this sandbox. Active-parameter count (~3B) does not mean only 3B weights need to be resident: every routed expert may be selected. Measure the five-step probe with Xcode/browser load reduced before any full run.

### Which modules to adapt

| Module family | First pilot choice | Reason / risk |
|---|---|---|
| Full attention `self_attn.q_proj/k_proj/v_proj/o_proj` | Adapt in the one full-attention block among the last four | Low additional count; covers the full-attention path without changing embeddings, norms, or output head. |
| Gated DeltaNet `linear_attn.in_proj_qkv/z/b/a`, `out_proj` | Adapt in the three linear-attention blocks among the last four | These projections carry most token mixing in this 3:1 architecture. Current `lora.yaml` lists these module names. |
| `mlp.shared_expert.{gate_proj,up_proj,down_proj}` and `mlp.shared_expert_gate` | Adapt in all four selected blocks | Shared path always runs and is only 311K rank-8 params across those blocks. |
| `mlp.switch_mlp.{gate_proj,up_proj,down_proj}` (routed experts) | Freeze initially | Every selected layer expands to 256 rank-specific adapters; even last four costs 62.9M parameters at rank 8 and alters many rarely used routes. Consider only a targeted subset after target-model routing traces and a separate memory probe. |
| `mlp.gate` router | Freeze | Top-k expert identity is a discrete routing decision and the target set is small. Router shifts can redirect tokens away from useful specialists; monitor route entropy and expert-token counts if ever unfrozen. |
| Embeddings, norms, output head | Freeze | Large/low-value change for this narrow task; increases forgetting risk. |

**Evidence boundary for expert specialization:** `results/20261104T185302Z-topic-study/metrics.json` and `results/20261104T192236Z-topic-study/metrics.json` report coding-route overlap above a mixed-set baseline (for the latter, coding leave-one-out mean ~0.24 vs mixed baseline ~0.044 at a 4 GB budget; 13 coding traces vs 3 non-coding traces). Those files report **48 layers**, while the target config has 40. They suggest topic-specialized routes in the traced model but do not identify the target Qwen3.6-35B experts. Collect target-model traces before selecting routed experts.

MoE load-balancing auxiliaries such as Switch Transformer's use both average router probability (P_i) and the fraction (f_i) of token assignments routed to expert (i):

\[
L_{aux}=\lambda N\sum_{i=1}^{N} f_i P_i,
\quad N=256,
\quad P_i=\frac1T\sum_t p_{t,i},
\quad f_i=\frac{\#\{(t,k):e_{t,k}=i\}}{kT}.
\]

See [Switch Transformers](https://arxiv.org/abs/2101.03961) and [GShard](https://arxiv.org/abs/2006.16668). This is a balancing pressure, not a drop-in guarantee against collapse. The installed MLX trainer optimizes next-token CE only; it does not add this auxiliary loss. Freezing the router is the practical way to avoid changing route balance in the first pilot. If the router is later unfrozen, add an auxiliary term and compare route counts/entropy to the base; do not assume the current `mlx_lm.lora` objective handles it.

Related adapter-specialization work includes [X-LoRA](https://arxiv.org/abs/2402.07148) and [Mixture of LoRA Experts (MoLE)](https://arxiv.org/abs/2404.13628). These papers study mixtures/routing of adapter experts; that is not direct evidence that fine-tuning this model's pretrained routed experts or router will improve pass rate.

## Preference optimization on pass/fail pairs

Let (y_w) be a test-passing solution and (y_l) a test-failing solution for the same prompt (x). The labels must come from the same deterministic task suite, not from an LLM judge. Compute completion log-probabilities over answer tokens only.

**DPO.** With a frozen reference policy πref and temperature β>0,

\[
\mathcal L_{DPO}=-\mathbb E\log\sigma\!\left(\beta\left[
\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}\right]\right).
\]

This is [Direct Preference Optimization](https://arxiv.org/abs/2305.18290): it compares chosen and rejected likelihood ratios to a reference. A conservative β around 0.1 is a starting hypothesis, not a validated setting here. The reference log-probs can be cached before training, avoiding a permanently resident second copy, but this still requires an extra reference scoring pass and custom preference batching.

**IPO.** [Identity Preference Optimization](https://arxiv.org/abs/2310.12036) replaces the logistic preference loss with a squared target margin:

\[
\mathcal L_{IPO}=\mathbb E\left[
\left(\log\frac{\pi_\theta(y_w\mid x)}{\pi_{ref}(y_w\mid x)}-
\log\frac{\pi_\theta(y_l\mid x)}{\pi_{ref}(y_l\mid x)}-
\frac{1}{2\beta}\right)^2\right].
\]

It may reduce over-optimization of pair labels, but still needs a reference and enough distinct pairs. It is not the first option for three pairs.

**ORPO.** [Odds-Ratio Preference Optimization](https://arxiv.org/abs/2403.07691) combines chosen-answer SFT with a reference-free odds-ratio preference term:

\[
\mathcal L_{ORPO}=\mathcal L_{SFT}(x,y_w)+\lambda\left[-\log\sigma\!\left(
\log\frac{\operatorname{odds}_\theta(y_w\mid x)}
{\operatorname{odds}_\theta(y_l\mid x)}\right)\right],
\qquad \operatorname{odds}_\theta(y\mid x)=\frac{P_\theta(y\mid x)}{1-P_\theta(y\mid x)}.
\]

No separate reference model is loaded, reducing memory and bookkeeping. The SFT term anchors the chosen solution. Sequence-length treatment and λ matter in implementation, so use the original paper's completion scoring rather than improvised token sums. For this project it is the best *future* first preference experiment because a real passing/failing pair is directly aligned to the test gate and is naturally represented by the current preference sidecar. Three pairs are far too few: wait for at least 50 train-only, distinct prompt/pair examples.

**KTO.** [Kahneman-Tversky Optimization](https://arxiv.org/abs/2402.01306) accepts independent desirable/undesirable labels rather than matched pairs. Define (r_\theta=\beta\log(\pi_\theta(y\mid x)/\pi_{ref}(y\mid x))), reference baseline (z_{ref}), and desirable label (s\in\{+1,-1\}):

\[
L_{KTO}(x,y,s)=
\begin{cases}
\lambda_+\,[1-\sigma(r_\theta-z_{ref})], & s=+1,\\
\lambda_-\,[1-\sigma(z_{ref}-r_\theta)], & s=-1.
\end{cases}
\]

This is useful if abundant standalone pass/fail labels exist, but discards the direct within-prompt comparison already available here and still uses a reference baseline. It is not a fit for three pairs.

**Installed support:** source audit of `mlx_lm/lora.py`, `mlx_lm/tuner/trainer.py`, `datasets.py`, and `losses.py` found no DPO/IPO/ORPO/KTO implementation or preference CLI in 0.32.2. The generic loss helpers include KL/JS functions, but the default trainer does not wire them into its CE objective. No MLX preference package/version is installed in this `.venv`; do not call a PyTorch-only package MLX-compatible. A future trainer needs its own version, formula, masking, and memory audit before use.

## Rejection sampling, ReST/STaR, and GRPO

**Rejection-sampling SFT / ReST-like loop.** For each train prompt, sample (K) answers from a fixed teacher/policy, execute the authoritative tests, reject failures, deduplicate passing outputs, then SFT on the accepted completions. A simplified dataset update is

\[
D_{t+1}=D_t\cup\{(x,y):y\sim\pi_t(\cdot\mid x),\;R_{tests}(x,y)=1\},
\qquad \pi_{t+1}=\operatorname{SFT}(D_{t+1}).
\]

[ReST](https://arxiv.org/abs/2308.08998) iterates generate/select/train with reward-based filtering; [STaR](https://arxiv.org/abs/2203.14465) iteratively generates rationales/answers and trains on successful examples. For this use, executable C/Python tests are a stronger acceptance signal than self-critique. The current builder's teacher loop executes the **teacher-provided** tests; because one model can make a weak test and wrong solution agree, manually/independently audit or add hidden assertions before treating the 250 planned teacher rows as fully verified. This generation/filter step needs no backward pass and is much more feasible than RL, though it still requires inference memory and repeated test runs. Never train on the held-out task prompts.

**GRPO / RL with verifiable rewards.** [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300) samples (G) completions (o_i) per prompt and compares their rewards within the group:

\[
A_i=\frac{R_i-\operatorname{mean}_{j=1}^{G}R_j}
{\operatorname{std}_{j=1}^{G}(R_j)+\epsilon}.
\]

A simplified clipped objective with a KL penalty is

\[
J(\theta)=\mathbb E\!\left[\frac1G\sum_{i=1}^{G}\frac1{|o_i|}\sum_t
\min\left(\rho_{i,t}A_i,\operatorname{clip}(\rho_{i,t},1-\epsilon_c,1+\epsilon_c)A_i\right)
-\beta D_{KL}(\pi_\theta\Vert\pi_{ref})\right],
\quad \rho_{i,t}=\frac{\pi_\theta(o_{i,t}\mid x,o_{i,<t})}
{\pi_{old}(o_{i,t}\mid x,o_{i,<t})}.
\]

With a 19 GB quantized base, only 12–16 GB realistically free while other applications are open, long code contexts, multiple rollouts, old/reference probabilities, optimizer states, and backward activations, GRPO is **not feasible to recommend on this M1 Max**. Installed MLX-LM also has no GRPO loop. Start with rejection-sampling data curation; RL would require an unmeasured, substantially different memory design and trainer.

## Quantization, adapter serving, and fusion

Serve the quantized 4-bit affine base with its small adapter applied as a sidecar. This avoids rewriting all base tensors and makes rollback simple. In MLX, the LoRA layer computes the quantized base linear plus the low-rank branch. `mlx_lm.fuse` can instead dequantize each quantized matrix, add ΔW, and by default re-quantize with the base matrix's original group size, bits, and mode. A second affine quantization changes the effective weights; the result is not bit-identical to “base quantized + adapter.” Fusing with `--dequantize` avoids the final quantization but emits a much larger full-precision model. The source implementation proves this code path exists; no numerical error was measured for this model. Keep the adapter separate for the pilot and compare held-out fixture results before considering fusion.

## Catastrophic forgetting and regularization

There is no portable “forgetting rate” that predicts this model's score change: it depends on data distribution, learning rate, number of steps, target modules, rank, and evaluation suite. Measure paired deltas on the frozen held-out tasks and the public mini. [InstructGPT](https://arxiv.org/abs/2203.02155) uses KL control against the reference policy during policy optimization; [LIMA](https://arxiv.org/abs/2305.11206) shows that a small, carefully curated SFT set can be effective. Neither gives a universal rate for a quantized 35B MoE adapter.

Practical controls here:

1. Keep the router, routed experts, embeddings, norms, and output head frozen; train only four late blocks.
2. Use a small LR (`1e-5` in the checked-in recipe), batch 1, three epochs maximum for the proposed ~231 train rows, and stop if validation loss or held-out performance turns down.
3. Keep a small general-code replay fraction (about 5–10%) only if it is license-cleared, task-disjoint, and itself tested; do not add generic conversational data just to inflate size.
4. KL-to-base regularization is an option for custom preference/RL training, but the installed SFT trainer does not include a KL term. Do not claim that it is active in the present SFT command.
5. Gate on first-attempt pass rate, pass-at-cap, review-note acceptance, and public-mini regression, not training loss alone.

## Primary source and local evidence index

- [InstructGPT](https://arxiv.org/abs/2203.02155): supervised completion objective and reference-policy KL control.
- [LoRA](https://arxiv.org/abs/2106.09685), [DoRA](https://arxiv.org/abs/2402.09353), [QLoRA](https://arxiv.org/abs/2305.14314): adapter math and quantized-base methods.
- [DPO](https://arxiv.org/abs/2305.18290), [IPO](https://arxiv.org/abs/2310.12036), [ORPO](https://arxiv.org/abs/2403.07691), [KTO](https://arxiv.org/abs/2402.01306): preference objectives.
- [ReST](https://arxiv.org/abs/2308.08998), [STaR](https://arxiv.org/abs/2203.14465), [DeepSeekMath/GRPO](https://arxiv.org/abs/2402.03300): self-training and verifiable-reward RL.
- [Switch Transformers](https://arxiv.org/abs/2101.03961), [GShard](https://arxiv.org/abs/2006.16668): MoE routing and load-balancing.
- [MLX-LM 0.32.2 official source](https://github.com/ml-explore/mlx-lm/tree/v0.32.2): [`lora.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/lora.py), [`tuner/lora.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/lora.py), [`tuner/dora.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/dora.py), [`tuner/datasets.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/datasets.py), [`tuner/trainer.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/tuner/trainer.py), [`models/qwen3_5.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_5.py), [`models/qwen3_next.py`](https://github.com/ml-explore/mlx-lm/blob/v0.32.2/mlx_lm/models/qwen3_next.py).
- Local records: `.venv` package metadata says `mlx-lm 0.32.2` and `mlx 0.32.4`; the model config supplies the dimensions above; `results/20261104T185302Z-topic-study/metrics.json` and `results/20261104T192236Z-topic-study/metrics.json` supply the routing-study summary; `docs/finetune/lora.yaml` and `RECIPE.md` supply current settings and the incomplete memory probe.
