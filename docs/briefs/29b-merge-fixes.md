<task>
Fix-up of tools/merge_task_vector.py (your previous run, brief docs/briefs/29-merge-tool.md). The orchestrator ran
tools/test_merge_vs_mlx.py on the Mac: dequantize OK path aside, the PACKED q differs from mx.quantize in 92% of
elements (4-bit). Three changes; change nothing else.
</task>

<fix_1_mlx_affine_quantize>
MLX's affine quantization (verified on the Mac by probing mx.quantize, group 64) is, per group, n_bins = 2^bits - 1:
    scale = max((wmax - wmin) / n_bins, eps)
    if |wmin| > |wmax|: edge = wmin            (scale stays positive)
    else:               edge = wmax; scale = -scale
    q0 = round(edge / scale)
    if q0 != 0: scale = edge / q0              (makes the edge value exactly representable)
    bias = edge  (bias = 0 if q0 == 0)
    q = clip(round((w - bias) / scale), 0, n_bins)
Probe results to match exactly: group of [-1.0, -0.2, 0.3, 0.5]*16 -> scale 0.1, bias -1.0;
group of [-0.2, 0.1, 0.3, 1.0]*16 -> scale -0.083333336, bias 1.0. Note round() is round-half-to-even (np.rint),
and the computation is float32. Make tools/test_merge_vs_mlx.py's assertions the acceptance criteria; also add these
two probe groups to tools/test_merge_task_vector.py as numpy-only checks.
</fix_1_mlx_affine_quantize>

<fix_2_text_only>
The base checkpoint also contains 333 vision-tower tensors that KAT lacks. Add --text-only: merge only tensors whose
logical names exist in BOTH checkpoints under the language-model prefix; copy every base-only tensor (vision tower)
unchanged; still refuse any mismatch inside the language model. Record the copied keys count in merge_meta.json.
</fix_2_text_only>

<fix_3_tokenizer_check>
Replace the tokenizer.json SHA-256 refusal with a semantic check: identical vocab (token -> id map), identical
added/special tokens and EOS ids, identical chat_template. (On the Mac these are identical for base vs KAT although
the file hashes differ.) Keep printing both hashes for the record. Implement with the tokenizers / json files only,
no mlx import.
</fix_3_tokenizer_check>

<verification_loop>
  .venv/bin/python tools/test_merge_task_vector.py
  .venv/bin/python tools/merge_task_vector.py --base ~/models/Qwen3.6-35B-A3B-4bit --finetune ~/models/KAT-Coder-V2.5-Dev-OptiQ-4bit --out /tmp/x --lam 0.5 --text-only --dry-run
Do NOT git add/commit. Report as before.
</verification_loop>
