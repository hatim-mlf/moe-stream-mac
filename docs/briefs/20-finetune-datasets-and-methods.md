<goal>
Research, from primary sources, (A) which public datasets we can use to fine-tune our local coding helper and
(B) the mathematics and practice of the fine-tuning methods that fit our situation. Write two documents:
docs/finetune/DATASETS.md and docs/finetune/METHODS.md, then update docs/finetune/README.md with a concrete,
justified pipeline. Research only: download nothing over 1 GB (small samples to inspect format are fine), do not
modify code, do not commit.
</goal>

<our_situation>
- Model: Qwen3.6-35B-A3B (MoE: 40 layers, 3 of 4 Gated DeltaNet linear-attention + 1 gated full attention,
  256 routed experts top-8 + 1 shared expert, ~3B active params), MLX 4-bit affine at ~/models/Qwen3.6-35B-A3B-4bit.
- Hardware: Apple M1 Max, 32 GB unified memory shared with Xcode/browsers (~12-16 GB realistically available),
  training must run locally with MLX (mlx-lm 0.32.2 is installed: read its source/--help for what it supports).
- Goal of the fine-tune: higher FIRST-ATTEMPT pass rate on precise, test-gated coding tasks (C99 kernels with
  fixtures, numpy/stdlib Python utilities, skeleton filling), and reliable compliance with review notes.
- What we already have (read them): docs/finetune/DIAGNOSIS.md (28 failed attempts classified: unused code/-Werror,
  spec drift, API/shape misuse, partially applied review notes, bounds/aliasing, signedness...), EVAL.md (frozen
  held-out split), DATA.md (28 verified seeds), RECIPE.md, tools/helper_rules.md.
</our_situation>

<part_A_datasets>
Use the camofox MCP browser for ALL web access, always with userId "moe-research" (older profiles fail a version
check; do not use or delete them). Start from raw sources: Hugging Face dataset pages + their API
(https://huggingface.co/api/datasets/<id>, ...?expand[]=cardData), dataset cards, the papers that introduced
them (arXiv), and their GitHub repos. Not blog lists.
Find and evaluate at least 12 candidates across these kinds:
- execution-verified code instruction data (solutions checked by tests), e.g. KodCode, OpenCodeInstruct,
  OpenCoder SFT, Magicoder OSS-Instruct, SelfCodeAlign, CodeFeedback, rStar-Coder, and newer 2026 ones;
- code-repair / feedback-following data (fix-from-error, review-comment application, multi-turn repair);
- preference data for code (pairs of passing vs failing solutions) for DPO/ORPO/KTO;
- C / systems / numerical code specifically (our weakest area), and Swift if any;
- data for following precise specifications / instructions.
For each: id + link, size, license (commercial use?), languages, how solutions were verified (tests run? by whom?),
format (chat messages? fields), quality signals reported in the paper, and CONTAMINATION risk: does it overlap
HumanEval/MBPP/LiveCodeBench, and could it overlap our held-out tasks (check by keyword/function names from
tools/ft_eval_manifest.json). Recommend a concrete MIX (which datasets, how many examples each, filters) sized for
a LoRA run on this Mac, plus how our own logs and teacher-generated verified tasks fit in.
</part_A_datasets>

<part_B_methods_math>
Write the mathematics precisely (LaTeX-style equations in Markdown), with the source paper for each:
1. SFT: token-level cross-entropy with prompt masking; why masking matters for our case.
2. LoRA / DoRA / QLoRA: W' = W + (alpha/r) B A, initialisation, rank/alpha choice, which modules to adapt in this
   architecture (attention q/k/v/o, Gated DeltaNet projections, shared expert, routed experts, router) and the
   parameter/memory cost of each for THIS model (compute the numbers from config.json).
3. Preference methods: DPO loss (with beta, reference model), IPO, ORPO (odds-ratio, no reference model, cheaper in
   memory), KTO (unpaired); which fits our small, paired failing/passing data best and why.
4. Rejection-sampling fine-tuning / ReST / STaR-style self-improvement with execution feedback, and RL with
   verifiable rewards (GRPO: group-relative advantage, KL term) - feasibility on a 32 GB Mac (be honest).
5. MoE-specific issues: should the router be frozen? load-balancing / auxiliary losses during fine-tuning, risk of
   router collapse, fine-tuning only the experts that fire on code (we have routing traces showing coding uses a
   distinct expert subset: see results/*topic-study*/metrics.json), expert-specialised LoRA papers.
6. Quantisation interplay: LoRA on a 4-bit affine base, then serving with the adapter vs fusing and re-quantising
   (error introduced), and how mlx-lm handles it.
7. Catastrophic forgetting: rates, regularisation (KL to base, replay of general data, small LR, few epochs).
For each method, state whether installed mlx-lm supports it (read the source: mlx_lm/lora.py, tuner/*) or which
MLX-compatible package does, with versions.
</part_B_methods_math>

<deliverable_README>
Update docs/finetune/README.md: the recommended pipeline in order (e.g. rules+autofix baseline -> SFT on a curated
verified mix with masking -> optional ORPO on our failing/passing pairs), each step with expected gain, memory
estimate for this Mac, the go/no-go measurement that decides the next step, and the exact commands.
</deliverable_README>

<report>
Reply with: the recommended dataset mix (one line each), the recommended method sequence with one-line reasons,
the 3 claims you are least sure about, and the files written.
</report>
