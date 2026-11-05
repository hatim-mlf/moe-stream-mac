#!/usr/bin/env python3
"""Mac-only parity gate for NumPy affine quantization and MLX."""

from __future__ import annotations

import numpy as np
import mlx.core as mx

import merge_task_vector as merge


def main() -> None:
    rng = np.random.default_rng(20261105)
    for bits in (4, 8):
        values = rng.normal(size=(17, 128)).astype(np.float32)
        packed_np, scales_np, biases_np = merge.quantize_affine(values, bits, 64)
        packed_mx, scales_mx, biases_mx = mx.quantize(
            mx.array(values), group_size=64, bits=bits, mode="affine"
        )
        mx.eval(packed_mx, scales_mx, biases_mx)
        packed_mx_np = np.asarray(packed_mx)
        scales_mx_np = np.asarray(scales_mx)
        biases_mx_np = np.asarray(biases_mx)
        np.testing.assert_array_equal(packed_np, packed_mx_np, err_msg=f"{bits}-bit packed q")

        dequant_np = merge.dequantize_affine(packed_np, scales_np, biases_np, bits, 64)
        dequant_mx = mx.dequantize(
            packed_mx,
            scales_mx,
            biases_mx,
            group_size=64,
            bits=bits,
            mode="affine",
        )
        mx.eval(dequant_mx)
        # MLX dequantizes with a fused multiply-add: allow float32 rounding noise, packed q must match exactly
        np.testing.assert_allclose(dequant_np, np.asarray(dequant_mx), rtol=0, atol=1e-6, err_msg=f"{bits}-bit dequantize")
        print(f"{bits}-bit affine parity: packed q and dequantized values match exactly")


if __name__ == "__main__":
    main()
