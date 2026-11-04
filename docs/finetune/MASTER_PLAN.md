# Master plan: making the local coding helper better

**Model:** Qwen3.6-35B-A3B, MLX 4-bit. **Baseline:** 2/8 held-out tasks pass on the first attempt (with the rules
prompt). **Constraint:** 32 GB Mac shared with Xcode etc.: forward passes fit, gradient training does not (~27 GB),
except overnight with apps closed.

**Roles:** Claude writes specs, tests and skeletons, reviews, runs GPU measurements and decides. **Codex** (gpt-6-luna)
researches (camofox), writes bigger pieces, and acts as the *teacher* (reference solutions). **Qwen** (local) writes
small scripts, acts as the *student* (its failing and passing attempts become data), and writes commit messages.

## Phase 0: a measurement we can trust (first, ~1 day)

Everything below is judged by this. 8 held-out tasks are too few for confident comparisons.

| Step | Owner | Gate |
|---|---|---|
| Expand the held-out set to ~30 tasks (C kernels, Python/NumPy, skeleton-fill), each with automatic tests, written by Codex and audited by Claude, never used for training | Codex + Claude | every task's reference solution passes; Qwen baseline measured |
| Baselines: first attempt; with the autofix loop (3 attempts); review-note compliance; a public mini benchmark (EvalPlus subset) for regressions | Claude runs `tools/ft_eval.py` | numbers saved in `results/finetune/` |
| Statistics: paired bootstrap 95% CI per comparison; temperature 0 plus 3 sampled runs | Codex (in ft_eval) | CI reported for every claim |

**Rule for every later phase:** a method counts only if it beats the baseline with a CI that excludes zero, beats a
**random control of the same size** (random direction, random merge, random correction), and does not lose more than
2 points on the public mini benchmark.

## Phase 1: zero data, zero training (pure linear algebra on the weights)

| Method | Math | Needs | Expected | Owner |
|---|---|---|---|---|
| **1a. Quantisation-error correction** (LoftQ-style) | per matrix: E = W_ref − Q(W); keep the best rank-r part of E by SVD (Eckart–Young); serve as a small adapter | a higher-precision copy: the 8-bit MLX 35B (38 GB) as reference | 4-bit behaves closer to 8-bit; gain on code uncertain, must be measured | Codex writes the SVD tool (layer by layer, fits RAM); Claude evaluates |
| **1b. Task-vector merging** (task arithmetic, TIES, DARE) | τ = W_finetune − W_base; W' = W + λ·τ, with sign-conflict resolution (TIES) or drop-and-rescale (DARE) | a public coding fine-tune with **exactly our base** (Codex is verifying KAT-Coder now) | borrows someone else's coding training for free | Codex writes the merge tool (dequantise → merge → requantise per layer); Claude sweeps λ |
| **1c. Router biasing toward coding experts** | add b·1[e ∈ coding set] to the router logits before top-k | a 35B coding routing profile (C engine traces, ~15 min) | small; cheap to test; risk: worse on non-code | Qwen writes the bias hook; Claude sweeps b |

Go/no-go after Phase 1: keep only what passes the rule above.

## Phase 2: a little data, still no training (the model's internal directions)

| Method | Math | Needs | Owner |
|---|---|---|---|
| **2a. Reproduce first:** the public Qwen3.5-0.8B weight-space experiment (rikiyat.github.io) | difference of means / ablation | small model download | Codex + Claude: our numbers must match theirs |
| **2b. Steering for review-note compliance and spec adherence** | linear representation hypothesis: d = mean(acts ∣ follows) − mean(acts ∣ ignores); steer (h + αd) at inference, or orthogonalise weights W ← W − (W d) dᵀ | ~50–200 contrast prompts (we have the real failing/passing attempts; Codex generates more); an activation hook in the hybrid engine | Codex writes the hook; Qwen generates student attempts; Claude evaluates with a random-direction control |
| 2c. Sparse-autoencoder features | sparse dictionary learning | only if a public SAE for Qwen exists | research only |

## Phase 3: inference-time (no weight changes)

Best-of-N with test verification (our autofix loop) plus format-constrained decoding. Already partly built; measure
attempts-to-pass and wall time. This is the safety net under everything else.

## Phase 4: real fine-tuning (overnight on the Mac, or cloud)

| Step | Math | Notes |
|---|---|---|
| Data: ~250 test-verified tasks | rejection sampling (keep only solutions that pass) | Codex = teacher, Qwen = student, tests decide; held-out set never included |
| Night 1: pipeline check on 31 seed examples | — | proves it fits (raised `iogpu.wired_limit_mb`, apps closed), produces an adapter, evaluates automatically |
| Night 2: SFT with LoRA | masked cross-entropy; adapters on attention + shared expert, router and routed experts frozen; rsLoRA scaling, PiSSA init (SVD), KL/EWC penalty against forgetting | judged by the Phase 0 rule |
| Later: ORPO / DPO on pass/fail pairs | closed-form preference loss (no reward model) | only once ≥50 real pairs exist |
| Not on this Mac: GRPO / RL | — | too much memory and compute |

## Phase 5: combine and ship

Stack the winners that compose (e.g. merge → quantisation correction → steering → autofix), re-evaluate the stack
(effects don't always add up), then document results and update `docs/REPORT.md` and the public repo.

## Order and timing (realistic)

1. **Day 1:** Phase 0 (expand held-out, baselines), 1c router bias (cheap), start the training dataset. **Night 1:** training pipeline check.
2. **Day 2:** 1b merging (if a same-base fine-tune exists) and 1a correction (8-bit reference download, 38 GB), 2a reproduction. **Night 2:** real SFT run.
3. **Day 3:** 2b steering pilot, Phase 3 measurements, Phase 5 combination and write-up.

Disk budget: ~100 GB free. The 8-bit 35B reference (38 GB) and one fine-tune for merging (22–70 GB) fit, one at a
time if the bf16 fine-tune is needed.
