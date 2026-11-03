"""MLX reference for the whole model: teacher-forced logits on a prompt + a greedy continuation.

  python tests/model_ref.py <model dir> <n_gen> <out.bin> [prompt]
Runs in MLX native bf16 (a float32 cast exceeds the GPU wired-memory limit on a 32 GB Mac).
Layout: int32 n_prompt, n_gen, vocab; int32 ids[n_prompt + n_gen]; float32 logits[(n_prompt + n_gen - 1)][vocab]
(logits row t = prediction after consuming ids[0..t]).
"""
import sys

import mlx.core as mx
import numpy as np
from mlx_lm import load

model_dir, n_gen, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
prompt = sys.argv[4] if len(sys.argv) > 4 else "The capital of Morocco is Rabat. The capital of France is"
model, tok = load(model_dir)
ids = tok.encode(prompt)
n_prompt = len(ids)
for _ in range(n_gen):                       # greedy, full re-run each step (simple and exact)
    logits = model(mx.array(ids)[None])[0, -1]
    ids.append(int(mx.argmax(logits)))
all_logits = np.array(model(mx.array(ids[:-1])[None])[0].astype(mx.float32))
with open(out, "wb") as f:
    np.array([n_prompt, n_gen, all_logits.shape[-1]], "<i4").tofile(f)
    np.array(ids, "<i4").tofile(f)
    all_logits.astype("<f4").tofile(f)
print(f"prompt {n_prompt} tokens, generated {n_gen}: {tok.decode(ids[n_prompt:])!r} -> {out}")
