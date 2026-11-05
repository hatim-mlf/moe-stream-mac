"""Parity checks for the custom chunked GDN backward."""

from functools import partial
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gated_delta, qwen3_5

import gdn_vjp
from test_gdn_checkpoint import _make_tiny_moe, _model_loss


RECURRENCE_ATOL = 1e-5
GRADIENT_ATOL = 1e-4


def _make_inputs(
    timesteps: int, masked: bool, gate_ndim: int
) -> tuple[
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    mx.array,
    Optional[mx.array],
]:
    mx.random.seed(1000 + timesteps + 10 * gate_ndim)
    batch, key_heads, value_heads, key_dim, value_dim = 2, 2, 4, 3, 5
    q = mx.random.normal((batch, timesteps, key_heads, key_dim))
    k = mx.random.normal((batch, timesteps, key_heads, key_dim))
    q, k = gated_delta.normalize_qk(q, k, inv_scale=key_dim**-0.5, eps=1e-6)
    v = mx.random.normal((batch, timesteps, value_heads, value_dim))
    g_shape = (batch, timesteps, value_heads)
    if gate_ndim == 4:
        g_shape += (key_dim,)
    g = mx.random.uniform(low=0.7, high=1.0, shape=g_shape)
    beta = mx.random.uniform(
        low=0.05, high=0.95, shape=(batch, timesteps, value_heads)
    )
    state = mx.random.normal(
        (batch, value_heads, value_dim, key_dim), dtype=mx.float32
    )
    mask: Optional[mx.array] = None
    if masked:
        mask = mx.array(
            np.random.default_rng(timesteps + gate_ndim).random(
                (batch, timesteps)
            ) > 0.2
        )
    return q, k, v, g, beta, state, mask


def _reference(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
    mask: Optional[mx.array],
) -> tuple[mx.array, mx.array]:
    return gated_delta.gated_delta_ops(q, k, v, g, beta, state, mask)


def _actual(
    chunk: int, *primals: mx.array, mask: Optional[mx.array]
) -> tuple[mx.array, mx.array]:
    q, k, v, g, beta, state = primals
    return gdn_vjp.gated_delta_chunked(
        q, k, v, g, beta, state, mask, chunk, impl="staged"
    )


def _assert_kernel_parity(gate_ndim: int) -> None:
    mx.random.seed(777 + gate_ndim)
    batch, timesteps, key_heads, value_heads, dim, value_dim = 1, 2, 1, 2, 32, 3
    q = mx.random.normal((batch, timesteps, key_heads, dim))
    k = mx.random.normal((batch, timesteps, key_heads, dim))
    q, k = gated_delta.normalize_qk(q, k, inv_scale=dim**-0.5, eps=1e-6)
    v = mx.random.normal((batch, timesteps, value_heads, value_dim))
    g_shape = (batch, timesteps, value_heads)
    if gate_ndim == 4:
        g_shape += (dim,)
    g = mx.random.uniform(low=0.7, high=1.0, shape=g_shape)
    beta = mx.random.uniform(low=0.05, high=0.95, shape=g_shape[:3])
    state = mx.random.normal((batch, value_heads, value_dim, dim))
    mask = mx.array([[True, False]])
    primals = (q, k, v, g, beta, state)
    dy = mx.random.normal((batch, timesteps, value_heads, value_dim))
    dstate = mx.random.normal(state.shape)
    expected, expected_grads = mx.vjp(
        partial(_reference, mask=mask), primals, (dy, dstate)
    )
    kernel_fn = lambda *xs: gdn_vjp.gated_delta_chunked(
        *xs, mask=mask, chunk=timesteps, impl="kernel"
    )
    actual, actual_grads = mx.vjp(kernel_fn, primals, (dy, dstate))
    mx.eval(*expected, *actual, *expected_grads, *actual_grads)
    for left, right in zip((*expected, *expected_grads), (*actual, *actual_grads)):
        np.testing.assert_allclose(
            np.asarray(left), np.asarray(right), atol=GRADIENT_ATOL, rtol=0
        )


@pytest.mark.parametrize("timesteps", [1, 63, 64, 65, 200])
@pytest.mark.parametrize("chunk", [16, 64])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("gate_ndim", [3, 4])
def test_forward_and_six_cotangents_match_ops(
    timesteps: int, chunk: int, masked: bool, gate_ndim: int
) -> None:
    q, k, v, g, beta, state, mask = _make_inputs(timesteps, masked, gate_ndim)
    primals = (q, k, v, g, beta, state)
    dy = mx.random.normal((q.shape[0], timesteps, v.shape[2], v.shape[3]))
    dstate = mx.random.normal(state.shape, dtype=mx.float32)
    expected, expected_grads = mx.vjp(
        partial(_reference, mask=mask), primals, (dy, dstate)
    )
    actual, actual_grads = mx.vjp(
        partial(_actual, chunk, mask=mask), primals, (dy, dstate)
    )
    mx.eval(*expected, *actual, *expected_grads, *actual_grads)
    for expected_value, actual_value in zip(expected, actual):
        np.testing.assert_allclose(
            np.asarray(expected_value), np.asarray(actual_value),
            atol=RECURRENCE_ATOL, rtol=0
        )
    assert len(expected_grads) == len(actual_grads) == 6
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        np.testing.assert_allclose(
            np.asarray(expected_grad), np.asarray(actual_grad),
            atol=GRADIENT_ATOL, rtol=0
        )
    if (timesteps, chunk, masked, gate_ndim) == (1, 16, False, 3):
        _assert_kernel_parity(3)
        _assert_kernel_parity(4)


def _assert_tree_close(expected: object, actual: object) -> None:
    expected_items = tree_flatten(expected)
    actual_items = tree_flatten(actual)
    assert [key for key, _ in expected_items] == [key for key, _ in actual_items]
    for (_, expected_value), (_, actual_value) in zip(expected_items, actual_items):
        np.testing.assert_allclose(
            np.asarray(expected_value), np.asarray(actual_value),
            atol=GRADIENT_ATOL, rtol=0
        )


def test_install_matches_tiny_moe_loss_and_lora_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx.random.seed(321)
    baseline_update = gdn_vjp._ORIGINAL_UPDATE or qwen3_5.gated_delta_update
    monkeypatch.setattr(qwen3_5, "gated_delta_update", baseline_update)
    model = _make_tiny_moe()
    inputs = mx.array([[17, 42, 81, 109, 211]], dtype=mx.int32)
    targets = mx.array([[42, 81, 109, 211, 307]], dtype=mx.int32)
    value_grad = nn.value_and_grad(model, _model_loss)
    expected_loss, expected_grads = value_grad(model, inputs, targets)
    mx.eval(expected_loss, expected_grads)
    gdn_vjp.install()
    actual_loss, actual_grads = value_grad(model, inputs, targets)
    mx.eval(actual_loss, actual_grads)
    np.testing.assert_allclose(
        np.asarray(expected_loss), np.asarray(actual_loss),
        atol=GRADIENT_ATOL, rtol=0
    )
    _assert_tree_close(expected_grads, actual_grads)


def test_install_delegates_use_kernel_true_to_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    sentinel = object()

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("use_kernel"))
        return sentinel

    monkeypatch.setattr(gdn_vjp, "_ORIGINAL_UPDATE", None)
    monkeypatch.setattr(qwen3_5, "gated_delta_update", spy)
    gdn_vjp.install()
    values = [mx.array(0.0) for _ in range(7)]
    result = qwen3_5.gated_delta_update(*values, use_kernel=True)
    assert result is sentinel
    assert calls == [True]
