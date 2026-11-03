"""Fixtures for engine/include/ops.h, ground truth from MLX, with the header's spec re-checked in NumPy.

File layout: int32 n_arrays, then per array: int32 length, float32 values.
"""
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "engine/tests/fixtures")
RNG = np.random.default_rng(7)


def save(name: str, *arrays: np.ndarray) -> None:
    with (OUT / f"ops_{name}.bin").open("wb") as f:
        np.array([len(arrays)], "<i4").tofile(f)
        for a in arrays:
            a = np.asarray(a, dtype="<f4").ravel()
            np.array([a.size], "<i4").tofile(f)
            a.tofile(f)


def check(name: str, got: np.ndarray, want: np.ndarray, tol: float = 1e-5) -> None:
    err = np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64)).max()
    assert err < tol, f"spec mismatch in {name}: {err}"
    print(f"spec ok  {name:8} (max err {err:.1e})")


def rmsnorm() -> None:
    n = 2048
    x = RNG.normal(size=n).astype(np.float32) * 3; w = RNG.normal(size=n).astype(np.float32)
    want = np.array(mx.fast.rms_norm(mx.array(x), mx.array(w), 1e-6))
    spec = x / np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-6) * w
    check("rmsnorm", spec, want); save("rmsnorm", x, w, want)


def silu_mul_sigmoid_softmax() -> None:
    n = 512
    g = RNG.normal(size=n).astype(np.float32) * 4; u = RNG.normal(size=n).astype(np.float32)
    want = np.array(nn.silu(mx.array(g)) * mx.array(u))
    check("silu_mul", g / (1 + np.exp(-g.astype(np.float64))) * u, want); save("silu_mul", g, u, want)
    want = np.array(mx.sigmoid(mx.array(g)))
    check("sigmoid", 1 / (1 + np.exp(-g.astype(np.float64))), want); save("sigmoid", g, want)
    logits = RNG.normal(size=256).astype(np.float32) * 6 + 40  # large offset: needs max-subtraction
    want = np.array(mx.softmax(mx.array(logits), precise=True))
    e = np.exp(logits.astype(np.float64) - logits.max())
    check("softmax", e / e.sum(), want, 1e-6); save("softmax", logits, want)


def topk() -> None:
    x = RNG.normal(size=256).astype(np.float32)
    k = 8
    want = np.argsort(-x, kind="stable")[:k]
    mlx_set = set(np.array(mx.argpartition(mx.array(x), kth=-k)[-k:]).tolist())
    assert mlx_set == set(want.tolist()), "topk set differs from MLX"
    print("spec ok  topk")
    save("topk", x, np.array([k]), want.astype(np.float32))


def rope() -> None:
    n_heads, head_dim, rot, theta = 4, 256, 64, 1e7
    for pos in (0, 7, 3001):
        x = RNG.normal(size=(n_heads, head_dim)).astype(np.float32)
        want = np.array(nn.RoPE(rot, traditional=False, base=theta)(mx.array(x)[None, :, None, :], offset=pos))[0, :, 0, :]
        spec = x.astype(np.float64).copy(); half = rot // 2
        a = pos * theta ** (-2 * np.arange(half) / rot)
        x1, x2 = spec[:, :half].copy(), spec[:, half:rot].copy()
        spec[:, :half] = x1 * np.cos(a) - x2 * np.sin(a); spec[:, half:rot] = x2 * np.cos(a) + x1 * np.sin(a)
        check(f"rope@{pos}", spec, want, 2e-4)
        save(f"rope_p{pos}", np.array([n_heads, head_dim, rot, pos, theta]), x, spec)  # spec: double-precision angle


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rmsnorm(); silu_mul_sigmoid_softmax(); topk(); rope()
