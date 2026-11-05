<task>
LoRA training of our Qwen3.6-35B-A3B (MLX 4-bit, model_type qwen3_5_moe, vocab 248,320, mlx-lm 0.32.2 in .venv)
runs out of GPU memory on a 32 GB Mac for rows longer than ~1-2k tokens. Cause: mlx_lm's
`tuner/trainer.py:default_loss` computes full-vocab logits for EVERY position (`logits = model(inputs)`), then
masks to the completion (`mask_prompt: true`). At L tokens that is L x 248,320 logits plus their fp32
cross-entropy and gradient: several GB per 1k tokens on top of ~19.5 GB of weights. Prompts are long, completions
short, so almost all of that is wasted.

Measured on the Mac (MLX peak, batch 1, 470-token row): 1 LoRA layer 19.48 GiB, 4 layers 25.14 GiB (weights on
disk 19 GB), so each adapted layer adds ~1.9 GiB in the backward pass (plausibly quantized-expert dequantisation in
the MoE vjp). That per-layer cost is NOT your task; it is handled by training fewer layers. Your task is the
length-dependent part: 4-layer runs OOM at 1-2k tokens, and even 2-layer runs OOM on the first step at 1.2-3k
tokens, while forward-only validation at 9,153 tokens fits. 1 layer: 19.48 GiB at 470 tokens, 24.19 GiB at ~3k
tokens (~1.9 GiB per extra 1k tokens). Report the expected saving per 1k tokens from your change.

Write `tools/ft_lowmem_loss.py` providing `completion_only_loss(model, batch, lengths)` with EXACTLY the same
signature, return value `(ce, ntoks)` and masking semantics as `default_loss` (read it: steps 1..T,
mask = steps >= lengths[:,0] and steps <= lengths[:,1]), but which:
1. runs the decoder backbone only (no LM head) to get final-norm hidden states [B, T, H]. Read
   `mlx_lm/models/qwen3_5_moe.py` (and whatever it wraps, e.g. qwen3_5 / qwen3_next / a language_model submodule)
   to find how `Model.__call__` produces logits: which attribute is the backbone, whether the head is `lm_head`
   or tied embeddings (`embed_tokens.as_linear`), and any logit soft-capping/scaling. Reproduce it exactly.
2. selects only masked positions (for B=1: a contiguous slice [start-1, end) of the shifted sequence; for B>1 fall
   back to processing per row, or to computing the head in chunks of CHUNK=512 positions over the masked span),
3. applies the head + cross_entropy on those positions only, in chunks of at most CHUNK positions, summing fp32
   losses, so peak logits memory is CHUNK x vocab regardless of L.
Also `tools/ft_lora_lowmem.py`: same as `tools/ft_lora_probe.py` (keep its timing/peak printing, wired limit and
MLX_CACHE_LIMIT_GB handling) but replaces the loss: mlx_lm.lora's `train_model` calls `train(..., loss=...)` or
uses `default_loss` (read lora.py and trainer.py to find the exact hook; patch the name the call actually
resolves, for both training and `evaluate`). Same CLI as `python -m mlx_lm.lora`.
</task>

<tests>
`tools/test_ft_lowmem_loss.py` (pytest; the lead runs it on the Mac, you cannot run MLX in your sandbox):
- build a tiny random-init qwen3_5_moe model through mlx_lm's own model class and a small config (copy the
  config keys from /Users/hatim/models/Qwen3.6-35B-A3B-4bit/config.json `text_config` but shrink: 2-4 layers
  covering both linear-attention and full-attention layer types, hidden 64, experts 4, top-k 2, vocab 1000;
  read the config parser to see which keys are required). Seed everything.
- for random batches with B=1 and B=2, several (prompt, completion) length pairs incl. completion longer than
  CHUNK (set CHUNK small in the test via a parameter) and prompt length 0: `completion_only_loss` value and
  ntoks equal `default_loss` (atol 1e-4 on loss), and gradients w.r.t. the trainable parameters after
  `linear_to_lora_layers` match (`nn.value_and_grad`, atol 1e-4).
- a test that the head is never applied to more than CHUNK positions at once (wrap/spy the head).
</tests>

<constraints>
- Do not modify anything under .venv, tools/ft_eval.py, tools/ft_build_dataset.py, docs/finetune/lora.yaml.
- Named constants, functions < 50 lines, type hints. No silent excepts.
- Do not commit. Report: files, how you found the backbone/head (file:line in mlx_lm), anything unverified.
</constraints>
