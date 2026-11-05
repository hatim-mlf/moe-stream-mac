<task>
Phase 2b: residual activation steering for spec fidelity and review-note application on our local
Qwen3.6-35B-A3B (MLX 4-bit, model_type qwen3_5_moe, 40 decoder layers, hidden 2048). Read first:
docs/research/training-free-methods.md section "Pilot 2 — residual steering" (follow it, except: skip the Llama-2
CAA reproduction), docs/finetune/DIAGNOSIS.md, docs/finetune/EVAL.md, tools/router_bias.py (copy its server-wrapping
pattern: patch mlx_lm.server.ModelProvider.load), tools/route_nll.py, tools/ft_eval.py (frozen SYSTEM prompt).
Two deliverables.
</task>

<part_a_contrast_set>
data/steering/pairs.jsonl: 50 NEW contrast pairs, each {"pair_id", "task_id", "category": "spec"|"review"|"api",
"split": "build"|"dev", "messages": [system, user], "preferred": str, "unpreferred": str}. The system message is
EXACTLY ft_eval.SYSTEM. "spec": a precise small coding spec; preferred implements it exactly; unpreferred drifts
(extra feature, invented semantics, changed signature, omitted error path). "review": user = spec + a previous
attempt + precise review notes; preferred applies every note and changes nothing else; unpreferred applies only
some notes or changes other lines. "api": exact API contract given; unpreferred misuses return values/shapes.
Both answers are one fenced code block (the format the model must produce). Mix C99 and Python ~50/50.
Each pair has a test under data/steering/tests/<task_id>/ that the preferred answer PASSES and the unpreferred FAILS;
add tools/steer_pairs_check.py that runs every test both ways and exits nonzero on any violation.
Split by task (not by answer): ~35 build, ~15 dev. Novel tasks only: nothing from evals/heldout-v2, the legacy
held-out prompts, or their paraphrases; extend tools/ft_leak_check.py to scan data/steering/ too.
</part_a_contrast_set>

<part_b_tools>
tools/steering.py (mlx, numpy, mlx_lm only):
- find_layers(model): model.language_model.model.layers (fallback model.model.layers); ValueError if none.
- capture_means(model, tokenizer, rows, layers) -> dict: teacher-forced forward over system+user+answer (chat
  template, enable_thinking=False, same prefix logic as route_nll.split_ids); for each listed layer, the mean of that
  decoder layer's OUTPUT residual over assistant answer positions only; accumulate online (sum + count), never keep
  full sequences. Implement by temporarily wrapping the layer objects' __call__ (instance-level wrapper objects
  replacing entries of the layers list, restored in a finally block).
- build_directions(pref_means, unpref_means) -> np.ndarray [n_layers, hidden] float32 (difference of means) plus
  per-layer residual RMS (saved for strength scaling).
- install_steering(model, layer, vector, alpha) / remove_steering(model): h' = h + alpha * vector added to that
  layer's output ONLY on decode steps (calls whose input sequence length is 1 after the prompt prefill), never on
  system/user prompt tokens. Exactly reversible.
- random_directions(hidden, n, seed, norm): seeded Gaussian directions with the given L2 norm.
- CLI: `steering.py capture --model M --pairs P --split build --layers 8,12,...,36 --out dirs.npz`;
  `steering.py serve --directions dirs.npz --layer L --alpha A [--random-seed S] [mlx_lm.server args]`.
tools/test_steering.py: tests on a TINY random-init qwen3_5_moe-style model built from mlx_lm's own classes (no
weights): zero alpha is bit-identical; steering changes outputs only on decode steps, by exactly alpha*vector at the
hooked layer output; capture means equal a brute-force numpy computation; remove restores everything; random
directions have the requested norm and are seed-deterministic; invalid inputs raise ValueError.
</part_b_tools>

<constraints>
Your sandbox has no Metal GPU and no localhost: MLX may not import. Write test_steering.py anyway (the orchestrator
runs it on the Mac); keep the numpy parts testable without MLX and test them. You CAN compile C and run
.venv/bin/python for part A. Do not touch engine/, evals/, results/, the manifests, or tools/helper_rules.md.
Download nothing. Do NOT git add/commit.
</constraints>

<verification_loop>
  .venv/bin/python tools/steer_pairs_check.py          # 50 pairs: all preferred pass, all unpreferred fail
  .venv/bin/python tools/ft_leak_check.py
  .venv/bin/python -c "import ast,sys; ast.parse(open('tools/steering.py').read())"
  .venv/bin/python tools/test_steering.py              # if MLX imports in your sandbox; else say so
</verification_loop>

<structured_output_contract>
1. Pair table summary (counts by category/language/split). 2. Files touched. 3. Gate outputs. 4. Open questions.
</structured_output_contract>
