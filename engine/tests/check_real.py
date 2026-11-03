"""Compare engine real_matvec output with mx.quantized_matmul on the same checkpoint tensors."""
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

MODEL = Path(sys.argv[1]).expanduser()
P = "language_model.model.layers"
CASES = [  # (tensor base, bits, expert)
    (f"{P}.0.linear_attn.in_proj_qkv", 4, -1),
    (f"{P}.0.linear_attn.out_proj", 4, -1),
    (f"{P}.0.mlp.gate", 8, -1),
    (f"{P}.3.self_attn.q_proj", 4, -1),
    (f"{P}.0.mlp.switch_mlp.gate_proj", 4, 17),
    (f"{P}.0.mlp.switch_mlp.down_proj", 4, 200),
]
GROUP = 64


def mlx_reference(base: str, bits: int, expert: int) -> np.ndarray:
    weights = {}
    for shard in MODEL.glob("*.safetensors"):
        part = mx.load(str(shard))
        if f"{base}.weight" in part:
            weights = {k: part[f"{base}.{k}"] for k in ("weight", "scales", "biases")}
            break
    w, s, b = (weights[k][expert] if expert >= 0 else weights[k] for k in ("weight", "scales", "biases"))
    cols = w.shape[-1] * 32 // bits
    x = mx.sin(0.37 * mx.arange(cols, dtype=mx.float32) + 1.0)
    y = mx.quantized_matmul(x[None], w, s.astype(mx.float32), b.astype(mx.float32),
                            transpose=True, group_size=GROUP, bits=bits)
    return np.array(y[0])


failed = False
for base, bits, expert in CASES:
    out = Path("build") / "real_y.f32"
    subprocess.run(["./build/real_matvec", str(MODEL), base, str(bits), str(GROUP), str(expert), str(out)], check=True,
                   stdout=subprocess.DEVNULL)
    got, want = np.fromfile(out, dtype="<f4"), mlx_reference(base, bits, expert)
    err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-6)
    ok = got.shape == want.shape and err < 1e-4
    failed |= not ok
    print(f"{'PASS' if ok else 'FAIL'}  {base.split('layers.')[1]}{f' [expert {expert}]' if expert >= 0 else ''}: "
          f"{got.shape[0]} rows, max rel err {err:.1e}")
print("REAL-WEIGHT CHECK FAILED" if failed else "ALL REAL-WEIGHT CHECKS PASSED")
sys.exit(failed)
