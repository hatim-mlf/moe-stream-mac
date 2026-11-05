<task>
Phase 1b: write tools/merge_task_vector.py, a memory-bounded task-vector merger for two MLX-quantized checkpoints of
the SAME architecture: base = ~/models/Qwen3.6-35B-A3B-4bit (affine, group 64; mostly 4-bit, some 8-bit tensors per
its config.json "quantization"), finetune = ~/models/KAT-Coder-V2.5-Dev-OptiQ-4bit (affine, group 64, mixed 4/8-bit
per-tensor overrides in config.json "quantization"). Read docs/research/training-free-methods.md "Merge mechanics,
quantization, and memory" and "Pilot 1" first; follow them. Only read config.json, model.safetensors.index.json and
safetensors HEADERS of the real models (they are 20 GB each; do not load whole files in tests).
</task>

<spec>
merged = base + lam * tau, tau = dequant(finetune) - dequant(base), per tensor, then requantized to the BASE
tensor's own quantization (bits, group_size) so the output is a drop-in replacement for the base directory.
- MLX affine format: weight is uint32-packed, values low-bits-first within each uint32 (32/bits values per word);
  per group g: w = scales[g] * q + biases[g]; "<name>.weight", "<name>.scales", "<name>.biases". Quantize with the
  same rule MLX uses (mx.quantize affine: per group, scale = (max - min) / (2^bits - 1), bias = min, q = round((w -
  bias) / scale), with MLX's edge-case handling); implement in numpy and document the formula.
- Non-quantized tensors (norms, A_log, dt_bias, conv1d, etc.): merge in float32, cast back to base dtype.
- Options: --lam FLOAT (global), --lam-attn / --lam-experts / --lam-shared / --lam-router (override by module path;
  router = "mlp.gate.", shared = "shared_expert"), --method linear|dare|ties, --dare-p, --ties-trim (keep top-k
  fraction by |tau| per tensor), --seed. (For a single task vector TIES = magnitude trim + rescale-free; document.)
- Refuse (ValueError) on any key-set or shape mismatch after dequantization, different num_hidden_layers/experts/
  vocab, or different tokenizer.json SHA-256 (print both hashes; --allow-tokenizer-mismatch to override).
- Memory: process one tensor at a time; stacked expert tensors ([256, out, in]) in chunks along the expert axis;
  peak RSS should stay under ~4 GB. Write output shards with the base index's shard layout, copy base config.json,
  tokenizer files and chat template; write merge_meta.json (args, per-module mean |tau|/|w| ratios, hashes).
- --dry-run: verify compatibility and print per-module tau statistics on a sample of tensors without writing.
- Disk: refuse to start unless free space >= base size + 2 GB.
</spec>

<tests>
tools/test_merge_task_vector.py (numpy + safetensors only, runs in your sandbox): synthetic tiny checkpoints written
to a temp dir in MLX format (4- and 8-bit, group 64, a stacked expert tensor, a float tensor): lam=0 reproduces base
tensors byte-identically (skip requantization when lam=0 or tau==0 for that tensor); lam=1 equals
quantize(dequant(ft)) under base's bits; quantize/dequantize round trip error <= scale/2 per element; DARE with p=0
equals linear; mismatched shapes/keys raise; per-module lam overrides apply only to matching paths.
Also tools/test_merge_vs_mlx.py that the ORCHESTRATOR runs on the Mac: compares your numpy quantize/dequantize with
mx.quantize/mx.dequantize on random matrices (4 and 8 bit) — must match exactly for dequantize and for packed q.
</tests>

<constraints>
No Metal and no localhost in your sandbox; no downloads. Do not touch engine/, evals/, data/, results/, manifests.
Do NOT git add/commit.
</constraints>

<verification_loop>
  .venv/bin/python tools/test_merge_task_vector.py
  .venv/bin/python tools/merge_task_vector.py --base ~/models/Qwen3.6-35B-A3B-4bit --finetune ~/models/KAT-Coder-V2.5-Dev-OptiQ-4bit --out /tmp/x --lam 0.5 --dry-run
  (the dry run must read only headers + a few sampled tensors; report its output)
</verification_loop>

<structured_output_contract>
1. What changed. 2. Files. 3. Gate outputs incl. the dry-run tau statistics and tokenizer hashes. 4. Open questions.
</structured_output_contract>
