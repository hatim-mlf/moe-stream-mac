"""Hybrid engine vs plain MLX (same bf16 math) vs the C engine (f32) on a long sequence (fits-in-RAM models only).

  python engine/hybrid/hybrid_vs_mlx.py [model dir]   (uses engine/build/long_out.bin from run_model)
"""
import gc, sys
from pathlib import Path
import numpy as np, mlx.core as mx
from mlx_lm import load
sys.path.insert(0, str(Path.home() / "kimi-k3-c-engine/engine/hybrid"))
M = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "models/Qwen3.6-35B-A3B-4bit"
raw = open(str(Path.home() / "kimi-k3-c-engine/engine/build/long_out.bin"), "rb").read()
np_, ng, V = np.frombuffer(raw[:12], "<i4"); n = np_ + ng
ids = np.frombuffer(raw, "<i4", n, 12).tolist()[:-1]
c_eng = np.frombuffer(raw, "<f4", (n - 1) * V, 12 + 4 * n).reshape(n - 1, V)
model, _ = load(str(M))
mlx_l = np.array(model(mx.array(ids)[None])[0].astype(mx.float32)); del model; gc.collect(); mx.clear_cache()
from hybrid import HybridModel
hm = HybridModel(M); hyb = np.array(hm.forward_all(ids))
def cmp(a, b):
    rel = np.linalg.norm(a - b, axis=1) / np.linalg.norm(b, axis=1); same = (a.argmax(1) == b.argmax(1))
    return f"median rel {np.median(rel):.1e}, max rel {rel.max():.1e}, argmax agree {same.sum()}/{len(same)}"
print("plain MLX bf16 vs C engine f32 :", cmp(mlx_l, c_eng))
print("hybrid        vs C engine f32 :", cmp(hyb, c_eng))
print("hybrid        vs plain MLX bf16:", cmp(hyb, mlx_l))
