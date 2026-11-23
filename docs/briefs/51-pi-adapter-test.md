# Brief 51: test the Qwen3.8-27B pi adapter on the M1 Max 32 GB

Status: planned (2026-11-23). Runs on the Mac (Phase 3). Do this before brief 50 (Strata): no big download.

## Why

A social-media post claims Qwen3.8-27B + the pi LoRA ("qwen3.8-pi", 4×A6000) matched Opus 5.5 on one real bug
fix (60/60 tests, ~93 s, ~95k tokens in / 4.6k out). It's one task with no repo, so it can't be rerun. We check
the adapter on our own frozen tasks instead.

Adapter: [harness-hub/Qwen3.8-27B-pi-adapted](https://huggingface.co/harness-hub/Qwen3.8-27B-pi-adapted), rank-32
LoRA on `Qwen/Qwen3.8-27B`, RL-trained for the [pi coding agent](https://github.com/badlogic/pi-mono), thinking
off. Published numbers disagree between sources (SWE-bench Pro 29.4→62.7 % vs 39.2→64.7 %): read the model
card and record what it says now.

## Steps

1. **Download the adapter only** (`tools/fetch_model.py harness-hub/Qwen3.8-27B-pi-adapted`, or the adapter
   files from it). Stop and ask the user if it's over 1 GB. Base model: the existing
   `~/models/Qwen3.8-27B-official-mlx-4bit`, no new download. Record the adapter's revision hash.
2. **Check the base matches**: `adapter_config.json` `base_model_name_or_path` must be `Qwen/Qwen3.8-27B`.
   Refuse if it's another base or another revision family. Record `r`, `lora_alpha`, `target_modules`,
   `use_rslora`, `use_dora` (refuse DoRA: not supported by the conversion below).
3. **Convert PEFT → mlx-lm adapter** (`tools/peft_to_mlx.py`, new; delegate per CLAUDE.md, ~60 lines):
   - keys `base_model.model.model.layers.N.<path>.lora_A.weight` → `model.layers.N.<path>.lora_a`, and the same
     for `lora_B` → `lora_b`; check the model's own key names first (`mlx_lm` loads Qwen3.8 as `qwen3_5`;
     multimodal checkpoints may nest under `language_model.`). Every adapter key must map to a real linear
     layer in the 4-bit model: refuse on any unmatched key.
   - transpose: PEFT `lora_A` is (r, in), mlx `lora_a` is (in, r); `lora_B` (out, r) → `lora_b` (r, out).
   - `adapter_config.json` for mlx-lm: `fine_tune_type: lora`, `num_layers` = all layers,
     `lora_parameters: {rank: r, scale: lora_alpha / r (or / sqrt(r) if use_rslora), dropout: 0, keys: [...]}`.
   - **Fixture test first** (rule): a tiny random linear layer, a random PEFT LoRA; output of
     `W x + (alpha/r) B A x` in numpy must equal mlx-lm's LoRALinear with the converted weights (rel err < 1e-5).
4. **Sanity check on real weights**: 5 short coding prompts, greedy, thinking off. Compare the first 64 tokens
   against the adapter's GGUF in llama.cpp (`llama-cli --lora`, same prompts) if a GGUF LoRA is published;
   otherwise against base-without-adapter (it must differ, and the answers must be sensible). Expect small drift
   from 4-bit; a mismatch in the first few tokens means the conversion is wrong.
5. **Serve**: add `HELPER=qwen-pi` to `scripts/local/serve-mlx.sh` (base 4-bit + `adapters/qwen38-pi-v1`),
   served name `qwen3.8-pi`.
6. **Eval A, one-shot** (comparable to old numbers): `tools/qwen38_code_compare.py add-model --name pi`, same 30
   cases, thinking off, 4096 tokens, temp 0. Baselines: base 11/30, RFT 14/30, KAT (see addendum).
7. **Eval B, agentic through pi** (what the adapter was trained for): pi pointed at
   `http://127.0.0.1:<port>/v1`, model `qwen3.8-pi`; each of the 30 problems in a fresh temp dir with the stub +
   visible tests, hidden tests scored afterwards by the existing scorer. Limit 30 min and 40 tool calls per
   problem. Run the same with the base model (no adapter) through pi, so the adapter's effect is isolated.
   Record per problem: pass, wall time, tool calls, tokens in/out.
8. **Optional reference**: the same 30 through Claude Code / Opus for the "matches Opus" question, if the user
   agrees to the cost.

## Measure with apps open; record

`results/<UTC>-pi-adapter/metrics.json`: adapter revision, conversion check results, Eval A pass/30 with
paired CI vs base and RFT, Eval B pass/30 base-vs-pi with paired CI, median wall time, tokens, tool calls,
decode and prompt tok/s, peak memory and swap. Log the delegation in `docs/delegation-log.md`.

## Expected

Decode ~16.5 tok/s (measured for the base 4-bit). Agentic tasks: minutes each, not ~90 s as on 4×A6000.
