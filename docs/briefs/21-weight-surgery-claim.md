<goal>
Find the ORIGINAL source of a viral claim and verify it, then judge whether the method can help our local model.
Write docs/research/weight-surgery-claim.md. Research only: no model downloads over 1 GB, no code changes, no commit.
</goal>

<the_claim>
A Facebook post by "Md Ismail Sojal" (screenshot: docs/research/img/weight-surgery-claim.png) says:
"Claude Opus 5.5 performed targeted weight surgery on Qwen3.5-4B": identified the neurons causing reasoning loops
and removed that direction, editing 2 neuron columns in layer 29 (0.00012% of weights), no gradients, no training
loop, zero human intervention. Results on 150 held-out math problems: baseline 59 correct (22-25 repetition loops),
random patch of the same shape 47-57, the edit 92-94 correct with 0 loops.
</the_claim>

<how_to_work>
- Use the camofox MCP browser for ALL web access with userId "moe-research-2" (fresh profile). Keep at most 5 tabs
  open: reuse tabs with navigate_and_snapshot instead of opening new ones (closing tabs needs approval you don't have).
- Trace the claim to its origin: search X/Twitter, GitHub, Hugging Face (models/spaces with edited Qwen3.5-4B
  weights), arXiv, Reddit (r/LocalLLaMA), Anthropic/Claude posts, using phrases like "weight surgery", "Qwen3.5-4B",
  "repetition loops", "layer 29", "0.00012%", "neuron columns". Find: the original author, code, edited weights,
  the eval set, the exact procedure.
- Verify: does the code/model exist and match the claim? Is the eval set public and held out? Check the arithmetic:
  loops fell by 22-25 but correct answers rose by 33-35; explain or flag it. Is the random control sound?
- Prior work to relate it to (find the actual papers): refusal-direction ablation / "abliteration" (Arditi et al.
  2024), representation engineering, activation steering, "repetition neurons" / induction-loop studies, ROME/MEMIT
  style editing. What does the literature say about fixing repetition loops and about side effects?
</how_to_work>

<relevance_to_us>
Our helper is Qwen3.6-35B-A3B (MoE, 40 layers: Gated DeltaNet + gated attention, 256 experts top-8 + shared expert),
MLX 4-bit. Training it locally failed (needs ~27 GB GPU memory). Its failures (docs/finetune/DIAGNOSIS.md) are spec
drift, invented semantics, partially applied review notes, bounds errors, unused code - not loops.
Answer concretely: (1) which of these failure types could plausibly be addressed by finding and removing/adding a
direction (e.g. "ignores review notes" vs "follows them"), and which cannot (missing knowledge); (2) the exact
procedure we would run: contrast-pair data we have or could make, which layer/residual stream to read, how to compute
the direction (difference of means, PCA), how to apply it (weight orthogonalisation vs inference-time steering) in an
MLX 4-bit MoE model, and how to evaluate with our frozen held-out tasks (tools/ft_eval.py) plus a random-direction
control; (3) memory: confirm it needs only forward passes (activations), which our engines can stream.
</relevance_to_us>

<report>
Reply with: whether the original source was found (links), whether the claim holds up, the arithmetic verdict, the
2-3 failure types worth trying a direction on, and the file written.
</report>
