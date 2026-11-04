"""Verify the hybrid engine against the C engine (which matches MLX float32), and measure it.

  python engine/hybrid/hybrid_check.py <model dir> <engine_out.bin from run_model>
"""
import resource
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from hybrid import HybridModel  # noqa: E402

REL_TOL = 1e-1   # hybrid runs bf16 activations on the GPU (MLX bf16 vs f32 measured 1-8%); argmax must match

model_dir, ref_path = Path(sys.argv[1]).expanduser(), sys.argv[2]
raw = open(ref_path, "rb").read()
n_prompt, n_gen, vocab = np.frombuffer(raw[:12], "<i4"); n = n_prompt + n_gen
ids = np.frombuffer(raw, "<i4", n, 12).tolist()
ref = np.frombuffer(raw, "<f4", (n - 1) * vocab, 12 + 4 * n).reshape(n - 1, vocab)

t0 = time.perf_counter(); hm = HybridModel(model_dir)
print(f"loaded trunk in {time.perf_counter() - t0:.1f}s | GPU active {mx.get_active_memory() / 1e9:.2f} GB")

# 1. whole sequence in ONE batch (the prefill path)
hm.stats.__init__(); t0 = time.perf_counter()
got = np.array(hm.forward_all(ids[:-1])); dt = time.perf_counter() - t0
fails = 0
for t in range(n - 1):
    rel = np.linalg.norm(got[t] - ref[t]) / np.linalg.norm(ref[t]); same = got[t].argmax() == ref[t].argmax()
    fails += not (same and rel < REL_TOL)
    if not (same and rel < REL_TOL): print(f"FAIL  pos {t}: rel {rel:.2e} argmax {'same' if same else 'DIFF'}")
print(f"batched {n - 1} tokens: {(n - 1) / dt:.1f} tok/s | experts read {hm.stats.bytes / 1e9:.1f} GB in {hm.stats.read_s:.1f}s "
      f"| {n - 1 - fails}/{n - 1} positions agree with the C engine")

# 2. prompt as a batch, then greedy token-by-token generation (the chat path)
hm.reset(); hm.stats.__init__(); t0 = time.perf_counter()
logits = hm.forward(ids[:n_prompt]); t_prompt = time.perf_counter() - t0
gen, t0 = [], time.perf_counter()
for _ in range(n_gen):
    tok = int(mx.argmax(logits)); gen.append(tok); logits = hm.forward([tok])
t_gen = time.perf_counter() - t0
match = sum(a == b for a, b in zip(gen, ids[n_prompt:]))
print(f"prompt {n_prompt} tok in {t_prompt:.2f}s ({n_prompt / t_prompt:.1f} tok/s) | generate {n_gen} tok: "
      f"{n_gen / t_gen:.2f} tok/s | experts read {hm.stats.bytes / 1e9:.1f} GB in {hm.stats.read_s:.1f}s")
print(f"greedy tokens identical to C engine: {match}/{n_gen} -> {hm.tokenizer.decode(gen)!r}")
print(f"peak GPU memory {mx.get_peak_memory() / 1e9:.2f} GB | peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9:.2f} GB")
print("HYBRID MATCHES" if fails == 0 and match == n_gen else "HYBRID CHECK FAILED")
