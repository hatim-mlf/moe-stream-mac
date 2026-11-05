"""Fixtures for engine/include/gdn.h: MLX ground truth, with gdn.h's spec re-checked in NumPy first."""
import os
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.gated_delta import _gated_delta_step_ops, compute_g, normalize_qk

import mem_guard

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "engine/tests/fixtures")
RNG = np.random.default_rng(11)
f32 = lambda shape, s=1.0: (RNG.normal(size=shape) * s).astype(np.float32)


def save(name, *arrays):
    with (OUT / f"gdn_{name}.bin").open("wb") as f:
        np.array([len(arrays)], "<i4").tofile(f)
        for a in arrays:
            a = np.asarray(a, "<f4").ravel(); np.array([a.size], "<i4").tofile(f); a.tofile(f)


def check(name, got, want, tol=1e-5):
    err = np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64)).max()
    assert err < tol, f"spec mismatch {name}: {err}"; print(f"spec ok  {name:10} (max err {err:.1e})")


def conv():
    C, K, T = 64, 4, 6
    w = f32((C, K, 1)); xs = f32((T, C))
    conv1d = nn.Conv1d(C, C, kernel_size=K, groups=C, bias=False, padding=0); conv1d.weight = mx.array(w)
    inp = mx.concatenate([mx.zeros((1, K - 1, C)), mx.array(xs)[None]], axis=1)   # as GatedDeltaNet does
    want = np.array(nn.silu(conv1d(inp)))[0]                                     # [T, C]
    state = np.zeros((K - 1, C)); spec = []
    for t in range(T):
        pre = (w[:, :K - 1, 0].T * state).sum(0) + w[:, K - 1, 0] * xs[t]
        spec.append(pre / (1 + np.exp(-pre))); state = np.vstack([state[1:], xs[t]])
    check("conv", np.array(spec), want); save("conv", [C, K, T], w, xs, want)


def norm_and_gates():
    Dk, H = 128, 4
    q, k = f32((1, 1, H, Dk)), f32((1, 1, H, Dk))
    qn, kn = (np.array(t) for t in normalize_qk(mx.array(q), mx.array(k), inv_scale=Dk ** -0.5, eps=1e-6))
    l2 = lambda v, s: s * v / np.sqrt((v.astype(np.float64) ** 2).sum(-1, keepdims=True) + 1e-6)
    check("l2norm q", l2(q, Dk ** -0.5), qn); check("l2norm k", l2(k, 1.0), kn)
    save("l2norm", [H, Dk], q, qn, k, kn)
    Hv = 32
    a, b, A_log, dt = f32(Hv, 3), f32(Hv, 3), f32(Hv, 1), f32(Hv, 1)
    a[0] = 30.0  # exercise the softplus large-input branch
    g = np.array(compute_g(mx.array(A_log), mx.array(a), mx.array(dt))); beta = np.array(mx.sigmoid(mx.array(b)))
    t = (a + dt).astype(np.float64); sp = np.where(t > 20, t, np.log1p(np.exp(np.minimum(t, 20))))
    check("gates g", np.exp(-np.exp(A_log.astype(np.float64)) * sp), g); check("gates b", 1 / (1 + np.exp(-b.astype(np.float64))), beta)
    save("gates", [Hv], a, b, A_log, dt, g, beta)


def recurrence():
    Hk, Hv, Dk, Dv, T = 2, 4, 16, 8, 5
    state = mx.zeros((1, Hv, Dv, Dk)); spec_state = np.zeros((Hv, Dv, Dk))
    arrays = [[Hk, Hv, Dk, Dv, T]]
    for t in range(T):
        q, k = f32((Hk, Dk), 0.3), f32((Hk, Dk), 0.3); v = f32((Hv, Dv))
        g = RNG.uniform(0.5, 1.0, Hv).astype(np.float32); beta = RNG.uniform(0, 1, Hv).astype(np.float32)
        rep = Hv // Hk
        y, state = _gated_delta_step_ops(mx.repeat(mx.array(q)[None], rep, -2), mx.repeat(mx.array(k)[None], rep, -2),
                                         mx.array(v)[None], mx.array(g)[None], mx.array(beta)[None], state)
        y = np.array(y)[0]
        ys = np.zeros((Hv, Dv))
        for h in range(Hv):
            kh = h // rep; S = spec_state[h] * g[h]
            d = (v[h] - S @ k[kh]) * beta[h]; S = S + np.outer(d, k[kh]); spec_state[h] = S; ys[h] = S @ q[kh]
        check(f"recur t{t}", ys, y)
        arrays += [q, k, v, g, beta, y]
    check("recur state", spec_state, np.array(state)[0])
    save("recurrence", *arrays, np.array(state)[0])


if __name__ == "__main__":
    _total_memory = mem_guard.total_memory_bytes()
    _headroom = mem_guard.headroom_bytes_from_env(os.environ)
    print(mem_guard.startup_limit_message(_total_memory, _headroom), flush=True)
    mem_guard.start(total_bytes=_total_memory, headroom_bytes=_headroom)
    OUT.mkdir(parents=True, exist_ok=True)
    conv(); norm_and_gates(); recurrence()
