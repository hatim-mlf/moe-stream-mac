"""Generate qmat fixtures with MLX as ground truth: random bf16 weights, mx.quantize, y = dequant(W) @ x.

Fixture layout (little-endian): int32 rows, cols, bits, group_size; then uint32 w[], uint16 scales[],
uint16 biases[], float32 x[cols], float32 y[rows].
"""
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np

CASES = ((64, 256, 4, 64), (64, 256, 8, 64), (512, 2048, 4, 64), (256, 2048, 8, 64), (96, 384, 4, 32))


def write_case(out_dir: Path, rows: int, cols: int, bits: int, group_size: int) -> Path:
    mx.random.seed(rows * 7 + cols + bits)
    w = (mx.random.normal((rows, cols)) * 0.05).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    x = mx.random.normal((cols,)).astype(mx.float32)
    # Dequantise in float32: mx.dequantize with bf16 scales rounds W to bf16, which is not what we test.
    w_deq = mx.dequantize(wq, scales.astype(mx.float32), biases.astype(mx.float32),
                          group_size=group_size, bits=bits)
    y = w_deq @ x
    path = out_dir / f"qmat_{rows}x{cols}_b{bits}_g{group_size}.bin"
    with path.open("wb") as f:
        np.array([rows, cols, bits, group_size], dtype="<i4").tofile(f)
        np.array(wq).astype("<u4").tofile(f)
        np.array(scales.view(mx.uint16)).astype("<u2").tofile(f)
        np.array(biases.view(mx.uint16)).astype("<u2").tofile(f)
        np.array(x).astype("<f4").tofile(f)
        np.array(y).astype("<f4").tofile(f)
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "engine/tests/fixtures")
    out.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        print(write_case(out, *case))
