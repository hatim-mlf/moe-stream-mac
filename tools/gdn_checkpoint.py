"""Activation-checkpointed ops recurrence for Qwen3.5 Gated DeltaNet."""

from collections.abc import Callable
from typing import Optional

import mlx.core as mx
from mlx_lm.models import gated_delta as _gated_delta


DEFAULT_CHUNK = 64
Array = mx.array
ChunkFunction = Callable[
    [Array, Array, Array, Array, Array, Array], tuple[Array, Array]
]
UpdateFunction = Callable[..., tuple[Array, Array]]

_ORIGINAL_UPDATE: Optional[UpdateFunction] = None


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


def _checkpoint_chunk(mask: Optional[Array]) -> ChunkFunction:
    def run(
        q: Array,
        k: Array,
        v: Array,
        g: Array,
        beta: Array,
        state: Array,
    ) -> tuple[Array, Array]:
        return _run_chunk(q, k, v, g, beta, state, mask)

    return mx.checkpoint(run)


def gated_delta_ops_checkpointed(
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Optional[Array] = None,
    mask: Optional[Array] = None,
    chunk: int = DEFAULT_CHUNK,
) -> tuple[Array, Array]:
    """Run the ops recurrence in checkpointed time chunks.

    Inputs follow ``mlx_lm.models.gated_delta.gated_delta_ops``.  Each
    checkpoint retains its entry and exit state, then recomputes that chunk
    during backward.
    """
    if chunk <= 0:
        raise ValueError("chunk must be a positive integer")

    batch, timesteps, key_heads, key_dim = q.shape
    value_heads, value_dim = v.shape[-2:]
    if state is None:
        state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)

    if (repeat_factor := value_heads // key_heads) > 1:
        q = mx.repeat(q, repeat_factor, -2)
        k = mx.repeat(k, repeat_factor, -2)

    outputs = []
    for start in range(0, timesteps, chunk):
        end = min(start + chunk, timesteps)
        chunk_mask = None if mask is None else mask[:, start:end]
        run = _checkpoint_chunk(chunk_mask)
        output, state = run(
            q[:, start:end],
            k[:, start:end],
            v[:, start:end],
            g[:, start:end],
            beta[:, start:end],
            state,
        )
        outputs.append(output)
    return mx.concatenate(outputs, axis=1), state


def _checkpointed_update(
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
            raise RuntimeError("gdn_checkpoint.install() has not been called")
        return _ORIGINAL_UPDATE(
            q,
            k,
            v,
            a,
            b,
            A_log,
            dt_bias,
            state,
            mask,
            use_kernel=use_kernel,
            lower_bound=lower_bound,
            allow_neg_eigval=allow_neg_eigval,
        )

    beta = mx.sigmoid(b)
    if allow_neg_eigval:
        beta = beta * 2.0
    if lower_bound is None:
        g = _gated_delta.compute_g(A_log, a, dt_bias)
    else:
        g = _gated_delta.compute_lower_bound_g(A_log, a, dt_bias, lower_bound)
    if state is None:
        batch, _, _, key_dim = q.shape
        value_heads, value_dim = v.shape[-2:]
        state = mx.zeros((batch, value_heads, value_dim, key_dim), dtype=mx.float32)
    return gated_delta_ops_checkpointed(q, k, v, g, beta, state, mask)


def install() -> None:
    """Patch Qwen3.5's imported update alias for its training path only."""
    global _ORIGINAL_UPDATE
    from mlx_lm.models import qwen3_5

    if qwen3_5.gated_delta_update is _checkpointed_update:
        return
    _ORIGINAL_UPDATE = qwen3_5.gated_delta_update
    qwen3_5.gated_delta_update = _checkpointed_update
