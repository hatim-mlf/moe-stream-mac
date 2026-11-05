"""Chunk-bounded custom VJP for the Qwen3.5 Gated DeltaNet recurrence.

For each step define ``A = S_prev * decay``, ``r = v - A @ k``,
``delta = beta * r``, ``S_tilde = A + k outer delta``, and
``y = S_tilde @ q``.  A mask selects ``S_t = S_tilde`` or ``S_prev``;
the output ``y`` always uses ``S_tilde``.

Given ``dy`` and the cotangent ``dS`` of the state after the step, let
``dS_tilde = dy outer q + mask * dS``.  The per-step adjoints are::

    dq       = S_tilde.T @ dy
    u        = dS_tilde @ k
    d_beta   = sum_value(u * r)
    dv       = beta * u
    e        = beta * u
    dA       = dS_tilde - e outer k
    dk       = sum_value(dS_tilde * delta - A * e)
    dS_prev  = dA * decay + (1 - mask) * dS
    dg       = sum_value(dA * S_prev)       # vector gate, reduce value axis
    dg       = sum_value_key(dA * S_prev)   # scalar gate, reduce both axes

Here ``mask`` is one to keep the candidate state and zero to keep the previous
state; an unmasked call uses one.  Backward recomputes each chunk from its
saved boundary, then traverses that chunk in reverse.  A data dependency on the later chunk's
state cotangent serializes the lazy recomputations.
"""

from collections.abc import Callable
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models import gated_delta as _gated_delta


DEFAULT_CHUNK = 64
Array = mx.array
UpdateFunction = Callable[..., tuple[Array, Array]]

_ORIGINAL_UPDATE: Optional[UpdateFunction] = None
_GDN_BWD_KERNEL = None

_GDN_BWD_SOURCE = r"""
    constexpr int N = Dk / 32;
    auto n = thread_position_in_grid.z;
    auto b = n / Hv;
    auto hv = n % Hv;
    auto hk = hv / (Hv / Hk);
    auto dv = thread_position_in_grid.y;
    auto lane = thread_position_in_threadgroup.x;
    float dstate[N];
    auto state_offset = (n * Dv + dv) * Dk;
    for (int j = 0; j < N; ++j) {
      dstate[j] = static_cast<float>(dstate_next[state_offset + N * lane + j]);
    }
    for (int t = T - 1; t >= 0; --t) {
      auto q_t = q + ((b * T + t) * Hk + hk) * Dk;
      auto k_t = k + ((b * T + t) * Hk + hk) * Dk;
      auto v_idx = (b * T + t) * Hv * Dv + hv * Dv + dv;
      auto gate_idx = (b * T + t) * Hv + hv;
      float prev[N];
      float a[N];
      float prefix_a[N];
      float ds_candidate[N];
      for (int j = 0; j < N; ++j) {
        int idx = N * lane + j;
        prev[j] = static_cast<float>(state_in[state_offset + idx]);
      }
      for (int r = 0; r < t; ++r) {
        auto k_r = k + ((b * T + r) * Hk + hk) * Dk;
        auto v_r = (b * T + r) * Hv * Dv + hv * Dv + dv;
        auto gate_r = (b * T + r) * Hv + hv;
        float prefix_kv = 0.0f;
        for (int j = 0; j < N; ++j) {
          int idx = N * lane + j;
          float gt;
          if (VectorG) {
            gt = static_cast<float>(g[gate_r * Dk + idx]);
          } else {
            gt = static_cast<float>(g[gate_r]);
          }
          prefix_a[j] = prev[j] * gt;
          prefix_kv += prefix_a[j] * static_cast<float>(k_r[idx]);
        }
        prefix_kv = simd_sum(prefix_kv);
        float prefix_delta =
            (static_cast<float>(v[v_r]) - prefix_kv) *
            static_cast<float>(beta[gate_r]);
        if (mask[b * T + r] != 0) {
          for (int j = 0; j < N; ++j) {
            int idx = N * lane + j;
            prev[j] = prefix_a[j] +
                      static_cast<float>(k_r[idx]) * prefix_delta;
          }
        }
      }
      float kv = 0.0f;
      for (int j = 0; j < N; ++j) {
        int idx = N * lane + j;
        float gt;
        if (VectorG) {
          gt = static_cast<float>(g[gate_idx * Dk + idx]);
        } else {
          gt = static_cast<float>(g[gate_idx]);
        }
        a[j] = prev[j] * gt;
        kv += a[j] * static_cast<float>(k_t[idx]);
      }
      kv = simd_sum(kv);
      float residual = static_cast<float>(v[v_idx]) - kv;
      float bt = static_cast<float>(beta[gate_idx]);
      float delta = residual * bt;
      bool keep = mask[b * T + t] != 0;
      float dy = static_cast<float>(dy_in[v_idx]);
      float ddelta_local = 0.0f;
      float dg_local = 0.0f;
      for (int j = 0; j < N; ++j) {
        int idx = N * lane + j;
        float cot = dy * static_cast<float>(q_t[idx]);
        if (keep) {
          cot += dstate[j];
        }
        ds_candidate[j] = cot;
        ddelta_local += cot * static_cast<float>(k_t[idx]);
      }
      float ddelta = simd_sum(ddelta_local);
      float e = ddelta * bt;
      float da[N];
      for (int j = 0; j < N; ++j) {
        int idx = N * lane + j;
        float gt;
        if (VectorG) {
          gt = static_cast<float>(g[gate_idx * Dk + idx]);
        } else {
          gt = static_cast<float>(g[gate_idx]);
        }
        da[j] = ds_candidate[j] - e * static_cast<float>(k_t[idx]);
        float candidate = a[j] + static_cast<float>(k_t[idx]) * delta;
        int grad_idx = gate_idx * Dk + idx;
        int qk_grad_idx = ((b * T + t) * Hv + hv) * Dk + idx;
        atomic_fetch_add_explicit(&d_q[qk_grad_idx], candidate * dy,
                                  memory_order_relaxed);
        atomic_fetch_add_explicit(
            &d_k[qk_grad_idx],
            ds_candidate[j] * delta - a[j] * e,
            memory_order_relaxed);
        if (VectorG) {
          atomic_fetch_add_explicit(
              &d_g[grad_idx], da[j] * prev[j], memory_order_relaxed);
        } else {
          dg_local += da[j] * prev[j];
        }
        dstate[j] = da[j] * gt + (keep ? 0.0f : dstate[j]);
      }
      float dg_total = simd_sum(dg_local);
      if (lane == 0) {
        int beta_idx = (b * T + t) * Hv + hv;
        atomic_store_explicit(&d_v[v_idx], e, memory_order_relaxed);
        atomic_fetch_add_explicit(&d_beta[beta_idx], e * residual,
                                  memory_order_relaxed);
        if (!VectorG) {
          atomic_fetch_add_explicit(&d_g[beta_idx], dg_total,
                                    memory_order_relaxed);
        }
      }
    }
    for (int j = 0; j < N; ++j) {
      int idx = N * lane + j;
      atomic_store_explicit(&d_state[state_offset + idx], dstate[j],
                            memory_order_relaxed);
    }
"""


def _decay(g: Array) -> Array:
    if g.ndim == 2:
        return g[..., None, None]
    if g.ndim == 3:
        return g[..., None, :]
    raise ValueError(f"Unsupported gating shape {g.shape}")


def _run_chunk(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
    mask: Optional[Array],
) -> tuple[Array, Array]:
    outputs = []
    for t in range(q.shape[1]):
        step_mask = None if mask is None else mask[:, t]
        output, state = _gated_delta._gated_delta_step_ops(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], state, step_mask
        )
        outputs.append(output)
    return mx.stack(outputs, axis=1), state


def _copy_boundary(state: Array) -> Array:
    return mx.stop_gradient(mx.positive(state))


def _forward(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
    mask: Optional[Array],
    chunk: int,
) -> tuple[Array, Array, Array]:
    batch, timesteps, key_heads, _ = q.shape
    value_heads = v.shape[2]
    repeat = value_heads // key_heads
    if repeat * key_heads != value_heads:
        raise ValueError("q/k head count must divide the value head count")
    if repeat > 1:
        q = mx.repeat(q, repeat, axis=-2)
        k = mx.repeat(k, repeat, axis=-2)
    outputs = []
    boundaries = [_copy_boundary(state)]
    for start in range(0, timesteps, chunk):
        end = min(start + chunk, timesteps)
        chunk_mask = None if mask is None else mask[:, start:end]
        output, state = _run_chunk(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g[:, start:end], beta[:, start:end], state, chunk_mask
        )
        outputs.append(output)
        boundaries.append(_copy_boundary(state))
    y = mx.concatenate(outputs, axis=1)
    return y, state, mx.stack(boundaries, axis=0)


def _step_parts(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
    mask: Optional[Array],
) -> tuple[Array, Array, Array, Array, Array]:
    a = state * _decay(g)
    residual = v - (a * k[..., None, :]).sum(axis=-1)
    delta = residual * beta[..., None]
    candidate = a + k[..., None, :] * delta[..., None]
    if mask is None:
        next_state = candidate
    else:
        selector = mask[:, None, None, None]
        next_state = mx.where(selector, candidate, state)
    return candidate, next_state, a, residual, delta


def _repeat_heads(q: Array, k: Array, value_heads: int) -> tuple[Array, Array]:
    repeat = value_heads // q.shape[2]
    if repeat > 1:
        return mx.repeat(q, repeat, axis=-2), mx.repeat(k, repeat, axis=-2)
    return q, k


def _collapse_heads(gradient: Array, key_heads: int, repeat: int) -> Array:
    if repeat == 1:
        return gradient
    batch, timesteps, _, dim = gradient.shape
    return gradient.reshape(batch, timesteps, key_heads, repeat, dim).sum(axis=3)


def _chunk_vjp(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    initial_state: Array,
    mask: Optional[Array],
    dy: Array,
    dstate: Array,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    dependency = mx.sum(dstate * 0.0)
    initial_state = initial_state + dependency.astype(initial_state.dtype)
    states = [initial_state]
    for t in range(q.shape[1]):
        step_mask = None if mask is None else mask[:, t]
        next_state = _step_parts(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], states[-1], step_mask
        )[1]
        states.append(next_state)
    dq, dk, dv, dg, dbeta = [], [], [], [], []
    for t in range(q.shape[1] - 1, -1, -1):
        step_mask = None if mask is None else mask[:, t]
        selector = 1.0 if step_mask is None else step_mask[:, None, None, None]
        candidate, _, a, residual, delta = _step_parts(
            q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t], states[t], step_mask
        )
        ds_candidate = dy[:, t, ..., None] * q[:, t, ..., None, :]
        ds_candidate = ds_candidate + selector * dstate
        dq.append((dy[:, t, ..., None] * candidate).sum(axis=-2))
        ddelta = (ds_candidate * k[:, t, ..., None, :]).sum(axis=-1)
        dbeta.append((ddelta * residual).sum(axis=-1))
        e = ddelta * beta[:, t, ..., None]
        dv.append(e)
        dk.append(
            (ds_candidate * delta[..., None] - a * e[..., None]).sum(axis=-2)
        )
        da = ds_candidate - e[..., None] * k[:, t, ..., None, :]
        if g.ndim == 3:
            dg.append((da * states[t]).sum(axis=(-2, -1)))
        else:
            dg.append((da * states[t]).sum(axis=-2))
        dstate = da * _decay(g[:, t]) + (1.0 - selector) * dstate
    return (
        mx.stack(dq[::-1], axis=1), mx.stack(dk[::-1], axis=1),
        mx.stack(dv[::-1], axis=1), mx.stack(dg[::-1], axis=1),
        mx.stack(dbeta[::-1], axis=1), dstate
    )


def _get_backward_kernel() -> Any:
    global _GDN_BWD_KERNEL
    if not mx.metal.is_available():
        raise RuntimeError("impl='kernel' requires an available Metal device")
    if _GDN_BWD_KERNEL is None:
        _GDN_BWD_KERNEL = mx.fast.metal_kernel(
            name="gdn_chunk_backward",
            input_names=[
                "q", "k", "v", "g", "beta", "state_in", "dy_in",
                "dstate_next", "mask", "T"
            ],
            output_names=["d_q", "d_k", "d_v", "d_g", "d_beta", "d_state"],
            source=_GDN_BWD_SOURCE,
            atomic_outputs=True,
        )
    return _GDN_BWD_KERNEL


def _run_backward_kernel(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
    dy: Array,
    dstate: Array,
    mask: Optional[Array],
) -> tuple[Array, Array, Array, Array, Array, Array]:
    batch, timesteps, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if mask is None:
        mask = mx.ones((batch, timesteps), dtype=mx.bool_)
    kernel = _get_backward_kernel()
    return kernel(
        inputs=[q, k, v, g, beta, state, dy, dstate, mask, timesteps],
        template=[
            ("InT", q.dtype), ("StT", state.dtype), ("Dk", key_dim),
            ("Dv", value_dim), ("Hk", key_heads), ("Hv", value_heads),
            ("VectorG", g.ndim == 4)
        ],
        output_shapes=[
            (batch, timesteps, value_heads, key_dim),
            (batch, timesteps, value_heads, key_dim),
            (batch, timesteps, value_heads, value_dim),
            g.shape,
            (batch, timesteps, value_heads),
            state.shape,
        ],
        output_dtypes=[mx.float32] * 6,
        grid=(32, value_dim, batch * value_heads),
        threadgroup=(32, 1, 1),
        init_value=0,
    )


def _backward_kernel(
    primals: tuple[Array, ...],
    cotangents: tuple[Array, ...],
    outputs: tuple[Array, ...],
    chunk: int,
) -> tuple[Optional[Array], ...]:
    """Use register-held state and recompute each prefix for reverse steps."""
    q, k, v, g, beta, state, mask = primals
    _, _, boundaries = outputs
    dy, dstate, _ = cotangents
    _, timesteps, key_heads, _ = q.shape
    value_heads = v.shape[2]
    if q.shape[-1] % 32:
        raise ValueError("impl='kernel' requires key_dim divisible by 32")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("impl='kernel' requires q, k, and v to share a dtype")
    repeat = value_heads // key_heads
    q_grads, k_grads, v_grads, g_grads, beta_grads = [], [], [], [], []
    starts = list(range(0, timesteps, chunk))
    mx.eval(dstate)
    for index in range(len(starts) - 1, -1, -1):
        start = starts[index]
        end = min(start + chunk, timesteps)
        chunk_mask = None if mask is None else mask[:, start:end]
        grads = _run_backward_kernel(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g[:, start:end], beta[:, start:end], boundaries[index],
            dy[:, start:end], dstate, chunk_mask
        )
        dq, dk, dv, dg, dbeta, dstate = grads
        mx.eval(dq, dk, dv, dg, dbeta, dstate)
        q_grads.append(_collapse_heads(dq, key_heads, repeat))
        k_grads.append(_collapse_heads(dk, key_heads, repeat))
        v_grads.append(dv)
        g_grads.append(dg)
        beta_grads.append(dbeta)
    return (
        mx.concatenate(q_grads[::-1], axis=1).astype(q.dtype),
        mx.concatenate(k_grads[::-1], axis=1).astype(k.dtype),
        mx.concatenate(v_grads[::-1], axis=1).astype(v.dtype),
        mx.concatenate(g_grads[::-1], axis=1).astype(g.dtype),
        mx.concatenate(beta_grads[::-1], axis=1).astype(beta.dtype),
        dstate.astype(state.dtype),
        None,
    )


def _backward(
    primals: tuple[Array, ...],
    cotangents: tuple[Array, ...],
    outputs: tuple[Array, ...],
    chunk: int,
    impl: str,
) -> tuple[Optional[Array], ...]:
    if impl == "kernel":
        return _backward_kernel(primals, cotangents, outputs, chunk)
    staged = impl == "staged"
    q, k, v, g, beta, state, mask = primals
    y, _, boundaries = outputs
    dy, dstate, _ = cotangents
    batch, timesteps, key_heads, key_dim = q.shape
    value_heads = v.shape[2]
    repeat = value_heads // key_heads
    q, k = _repeat_heads(q, k, value_heads)
    q_grads, k_grads, v_grads, g_grads, beta_grads = [], [], [], [], []
    if staged:
        mx.eval(dstate)
    starts = list(range(0, timesteps, chunk))
    for index in range(len(starts) - 1, -1, -1):
        start = starts[index]
        end = min(start + chunk, timesteps)
        chunk_mask = None if mask is None else mask[:, start:end]
        grads = _chunk_vjp(
            q[:, start:end], k[:, start:end], v[:, start:end],
            g[:, start:end], beta[:, start:end], boundaries[index], chunk_mask,
            dy[:, start:end], dstate
        )
        dq, dk, dv, dg, dbeta, dstate = grads
        if staged:
            mx.eval(dq, dk, dv, dg, dbeta, dstate)
        q_grads.append(dq)
        k_grads.append(dk)
        v_grads.append(dv)
        g_grads.append(dg)
        beta_grads.append(dbeta)
    dq = _collapse_heads(mx.concatenate(q_grads[::-1], axis=1), key_heads, repeat)
    dk = _collapse_heads(mx.concatenate(k_grads[::-1], axis=1), key_heads, repeat)
    return (
        dq.astype(q.dtype),
        dk.astype(k.dtype),
        mx.concatenate(v_grads[::-1], axis=1).astype(v.dtype),
        mx.concatenate(g_grads[::-1], axis=1).astype(g.dtype),
        mx.concatenate(beta_grads[::-1], axis=1).astype(beta.dtype),
        dstate.astype(state.dtype),
        None,
    )


def _make_custom_function(
    chunk: int, impl: str
) -> Callable[..., tuple[Array, Array, Array]]:
    @mx.custom_function
    def recurrence(
        q: Array,
        k: Array,
        v: Array,
        g: Array,
        beta: Array,
        state: Array,
        mask: Optional[Array],
    ) -> tuple[Array, Array, Array]:
        return _forward(q, k, v, g, beta, state, mask, chunk)

    @recurrence.vjp
    def recurrence_vjp(
        primals: tuple[Any, ...],
        cotangents: tuple[Array, ...],
        outputs: tuple[Array, ...],
    ) -> tuple[Optional[Array], ...]:
        return _backward(primals, cotangents, outputs, chunk, impl)

    return recurrence


def gated_delta_chunked(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Optional[Array] = None,
    mask: Optional[Array] = None,
    chunk: int = DEFAULT_CHUNK,
    impl: str = "staged",
) -> tuple[Array, Array]:
    """Run GDN with saved chunk boundaries and a hand-written reverse pass."""
    if chunk <= 0:
        raise ValueError("chunk must be a positive integer")
    if impl not in ("chunked", "staged", "kernel"):
        raise ValueError("impl must be 'chunked', 'staged', or 'kernel'")
    batch, _, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if state is None:
        state = mx.zeros(
            (batch, value_heads, value_dim, key_dim), dtype=mx.float32
        )
    y, final_state, _ = _make_custom_function(chunk, impl)(
        q, k, v, g, beta, state, mask
    )
    return y, final_state


def _vjp_update(
    q: Array,
    k: Array,
    v: Array,
    a: Array,
    b: Array,
    A_log: Array,
    dt_bias: Array,
    state: Optional[Array] = None,
    mask: Optional[Array] = None,
    *,
    use_kernel: bool = True,
    lower_bound: float | None = None,
    allow_neg_eigval: bool = False,
) -> tuple[Array, Array]:
    if use_kernel:
        if _ORIGINAL_UPDATE is None:
            raise RuntimeError("gdn_vjp.install() has not been called")
        return _ORIGINAL_UPDATE(
            q, k, v, a, b, A_log, dt_bias, state, mask,
            use_kernel=use_kernel, lower_bound=lower_bound,
            allow_neg_eigval=allow_neg_eigval
        )
    beta = mx.sigmoid(b)
    if allow_neg_eigval:
        beta = beta * 2.0
    if lower_bound is None:
        g = _gated_delta.compute_g(A_log, a, dt_bias)
    else:
        g = _gated_delta.compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    # Real Qwen GDN uses Dk=128; smaller test configurations use the ops VJP.
    impl = "kernel" if q.shape[-1] % 32 == 0 else "staged"
    return gated_delta_chunked(q, k, v, g, beta, state, mask, impl=impl)


def install() -> None:
    """Patch Qwen3.5 training calls while preserving its inference update."""
    global _ORIGINAL_UPDATE
    from mlx_lm.models import qwen3_5

    if qwen3_5.gated_delta_update is _vjp_update:
        return
    _ORIGINAL_UPDATE = qwen3_5.gated_delta_update
    qwen3_5.gated_delta_update = _vjp_update
