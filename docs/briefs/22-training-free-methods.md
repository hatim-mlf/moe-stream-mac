<goal>
Find, from primary sources, the methods that can make our local coding helper better WITHOUT gradient training on our
own data (we cannot train it locally: it needs ~27 GB of GPU memory), judge each one for our exact model, and design a
rigorous pilot for the best two. Write docs/research/training-free-methods.md. Research only: download nothing over
1 GB (config.json / index files / model cards are fine), no code changes, no commit.
</goal>

<definitions>
Be precise about what "without training data" means for each method and label it:
(a) no gradient updates at all; (b) needs only a small contrast/calibration set (tens-hundreds of prompts, no labels);
(c) reuses training someone else already did (e.g. weights of an existing fine-tune); (d) inference-time only
(decoding / sampling / verification). Several methods are in more than one class.
</definitions>

<our_model_and_constraints>
- Qwen3.6-35B-A3B, MLX 4-bit affine (~/models/Qwen3.6-35B-A3B-4bit). MoE: 40 layers (3 of 4 Gated DeltaNet linear
  attention + 1 gated full attention), 256 routed experts top-8 + 1 shared expert, ~3B active.
- Hardware: M1 Max 32 GB shared with Xcode etc. Forward passes are fine (our engines stream experts:
  docs/C_ENGINE.md, docs/MLX_ENGINE.md); full gradient training is not.
- Its weaknesses (docs/finetune/DIAGNOSIS.md, 28 classified failures): unused code/-Werror, spec drift and invented
  semantics, API/shape misuse, partially applied review notes, bounds/aliasing, signedness. Baseline: 2/8 held-out
  first-attempt passes (results/finetune/baseline-heldout-a1.json). Frozen eval: tools/ft_eval.py, EVAL.md.
- We have routing traces showing coding uses a distinct expert subset (results/*topic-study*/metrics.json,
  engine/hybrid/profiles/).
- Prior notes: docs/research/weight-surgery-claim.md (direction editing), docs/finetune/METHODS.md (trained methods).
</our_model_and_constraints>

<how_to_work>
Use the camofox MCP browser for ALL web access with userId "moe-research-2"; keep at most 5 tabs: reuse tabs with
navigate_and_snapshot (closing tabs needs approval you don't have). Start from RAW material: arXiv papers, their
official GitHub code, Hugging Face model cards / config.json / safetensors index files, mergekit docs - not blogs.
Cover at least these families, citing the primary paper + code for each:
1. Model merging / task arithmetic with EXISTING fine-tunes of the same base: task vectors, TIES, DARE, model stock,
   SLERP; tools: mergekit. Crucial check: which public coding fine-tunes share OUR exact base and tokenizer? Verify
   from config.json + model card + base_model metadata: mlx-community/KAT-Coder-V2.5-Dev-OptiQ-4bit (and its source
   Kwaipilot repo), any Qwen3.6-35B-A3B / Qwen3.5-35B-A3B coding fine-tunes, abliterated/merged variants on HF.
   Merging needs same-architecture, same-base weights in compatible precision: explain how to do it with 4-bit MLX
   weights (dequantise -> merge -> requantise) and the memory needed per tensor (can be done layer by layer?).
2. Activation steering / directional ablation / representation engineering / inference-time intervention (ITI),
   contrastive activation addition; tools (repeng, steering-vectors, TransformerLens equivalents) and how to hook
   the residual stream in MLX for this hybrid architecture.
3. Sparse autoencoder features for Qwen models (do any exist? Gemma Scope-like releases?).
4. Weight editing (ROME, MEMIT, AlphaEdit) - relevance for coding rules (probably low; say why).
5. MoE-specific training-free interventions: router logit biasing toward experts used by coding, expert pruning or
   merging (REAP, MC-SMoE, other 2025-2026 papers), "expert specialisation" findings; what happens to quality.
6. Inference-time methods: contrastive decoding / DoLa, self-consistency, best-of-N with execution verification
   (our tools/autofix.py is a form of this), constrained decoding / grammar, speculative decoding (speed only).
For each method give: what it changes, data class (a-d), evidence of gains on CODE (benchmark, numbers, source), known
side effects, MLX/4-bit/MoE feasibility on this Mac, effort, and which of our failure types it could plausibly fix.
</how_to_work>

<deliverable>
docs/research/training-free-methods.md:
1. A ranked table of methods for OUR case (expected gain on our failure types x feasibility x risk), every claim linked.
2. A rigorous pilot design for the top 2: exact steps, data needed (and whether we have it), how to implement in MLX,
   compute/memory estimate, and evaluation: frozen held-out tasks (tools/ft_eval.py) with paired bootstrap CI,
   a RANDOM control of the same size/shape (random direction, random-delta merge), a public mini benchmark to
   catch regressions, and a first REPRODUCTION step on a small public model with a published result before touching ours.
3. What will NOT work for us and why (one line each).
</deliverable>

<report>
Reply with: the top 3 methods (one line each with the expected gain and why), whether any public coding fine-tune
shares our exact base (yes/no + evidence), and the file written.
</report>
