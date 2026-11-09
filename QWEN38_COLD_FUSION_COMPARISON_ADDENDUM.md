# Cold Fusion MLX comparison addendum

This addendum extends the completed base-versus-RFT pilot without changing its frozen task set or its saved outputs. It adds the Apple MLX candidate requested on 2026-11-08 and compares all three candidates on the same 15 problems, two languages, prompts, tests, and decoding settings.

## Added candidate

- Model: `Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ4e-1M`
- Local checkpoint: `/Users/hatim/models/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-MLX-oQ4e-1M`
- Format: MLX-compatible safetensors, oQ4e mixed-precision 4-bit.
- Model card: <https://huggingface.co/Solstice-AI/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU-mlx-oQ4e-1M>
- The user-supplied NEO-CODER-MAX-MTP link is a GGUF repository. This MLX release is from the related DavidAU NM-DAU Cold Fusion lineage cited by that repository; it is a distinct MLX quantization/build, not the byte-identical GGUF artifact. Record the resolved revision and local shard inventory in the run manifest.

## Evaluation procedure

1. Preserve the original base and RFT outputs. Generate only the new candidate's 30 outputs, using the same frozen task prompts, per-case seeds, temperature, and token limit.
2. Load one local model at a time. Do not load the new candidate until the existing RFT server is stopped; stop and verify the new server before scoring.
3. Score the new outputs against the same frozen tests and sandbox. Add candidate outputs and per-test results to each existing problem folder.
4. Report each model's pass@1, paired comparisons for all three pairs, language and difficulty breakdowns, per-problem outcomes, and problem-clustered 95% bootstrap intervals.
5. Update the single project-root comparison report. Keep the original two-candidate analysis JSON as provenance and save a separate three-candidate analysis JSON.

The benchmark remains one-shot: no repair loop, plan-first prompt, checklist, retrieval, tools, or feedback. Evaluate those runtime changes later as separate conditions on fresh held-out tasks.

## Result (2026-11-08, Claude)

Run: `results/20261108T000646Z-qwen38-code-comparison` (`tools/qwen38_code_compare.py add-turbo`), 30 cases, one-shot, temp 0.

| Model | Pass | Median reply time | Output tokens | Decode speed | Hit 4096-token cap |
|---|---|---|---|---|---|
| Base 4-bit | 11/30 | 96.6 s | 57,281 | 16.7 tok/s | 9 |
| Base + RFT adapter | 14/30 | 83.2 s | 57,357 | 16.5 tok/s | 5 |
| TURBO Cold Fusion oQ4e | 10/30 | 28.6 s | 17,916 | 13.8 tok/s | 1 |

TURBO vs base -3.3 pts (CI [-23.3,+16.7], 3W/4L); vs RFT -13.3 pts (CI [-36.7,+6.7], 2W/6L).
"Turbo" means ~3x shorter answers, not faster decoding (decode is ~17% slower: mixed 4/5/6-bit layers).
Verdict: not adopted; the helper stays base + RFT adapter.

## KAT-Coder-V2.5-Dev OptiQ 4-bit (2026-11-09, Claude)

`add-model --name kat`, same 30 cases and protocol (thinking off, 4096 tokens, temp 0). qwen3_5_moe, 256 experts, ~4.5 bpw, 20 GB.

| Model | Pass | Median reply | Decode | Hit cap |
|---|---|---|---|---|
| Base 4-bit | 11/30 | 96.6 s | 16.7 tok/s | 9 |
| Base + RFT | 14/30 | 83.2 s | 16.5 tok/s | 5 |
| TURBO (deleted) | 10/30 | 28.6 s | 13.8 tok/s | 1 |
| KAT-Coder | 15/30 | 22.1 s | 47.8 tok/s | 8 |

KAT vs base +13.3 pts (CI [0,+26.7], 5W/1L); vs RFT +3.3 pts (CI [-16.7,+20.0], 3W/2L). Hard tier: KAT 3/10, others 1-2.
Same accuracy as base + RFT within noise, ~3x faster decode. 8 replies hit the 4096-token cap, so a larger budget may help.
Decision deferred to the fresh 60-problem set (Aider Polyglot, with the repair loop) against base + RFT.
