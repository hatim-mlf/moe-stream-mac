"""Parity checks for the checkpointed Qwen3.5 Gated DeltaNet recurrence."""

from functools import partial
from typing import Callable, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import gated_delta, qwen3_5
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
from mlx_lm.tuner.utils import linear_to_lora_layers

import gdn_checkpoint


Array = mx.array
InputPack = tuple[Array, Array, Array, Array, Array, Array, Optional[Array]]
RECURRENCE_ATOL = 1e-5
GRADIENT_ATOL = 1e-4
VALUE_HEADS = 32
KEY_HEADS = 16
HEAD_DIM = 128


def _make_inputs(
    timesteps: int, masked: bool
) -> InputPack:
    mx.random.seed(1000 + timesteps)
    batch = 1
    q = mx.random.normal(
        (batch, timesteps, KEY_HEADS, HEAD_DIM), dtype=mx.float32
    )
    k = mx.random.normal(
        (batch, timesteps, KEY_HEADS, HEAD_DIM), dtype=mx.float32
    )
    v = mx.random.normal(
        (batch, timesteps, VALUE_HEADS, HEAD_DIM), dtype=mx.float32
    )
    q, k = gated_delta.normalize_qk(
        q, k, inv_scale=HEAD_DIM**-0.5, eps=1e-6
    )
    g = mx.random.uniform(
        low=0.9, high=1.0, shape=(batch, timesteps, VALUE_HEADS)
    )
    beta = mx.random.uniform(
        low=0.05, high=0.95, shape=(batch, timesteps, VALUE_HEADS)
    )
    state = mx.random.normal(
        (batch, VALUE_HEADS, HEAD_DIM, HEAD_DIM), dtype=mx.float32
    )
    mask = None
    if masked:
        mask = mx.array(
            np.random.default_rng(1000 + timesteps).random((batch, timesteps)) > 0.15
        )
    return q, k, v, g, beta, state, mask


def _assert_forward_matches(inputs: InputPack, chunk: int) -> None:
    q, k, v, g, beta, state, mask = inputs
    expected = gated_delta.gated_delta_ops(q, k, v, g, beta, state, mask)
    actual = gdn_checkpoint.gated_delta_ops_checkpointed(
        q, k, v, g, beta, state, mask, chunk
    )
    mx.eval(*expected, *actual)
    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_allclose(
            np.asarray(expected_array),
            np.asarray(actual_array),
            atol=RECURRENCE_ATOL,
            rtol=0,
        )


def _assert_gradients_match(inputs: InputPack, chunk: int) -> None:
    q, k, v, g, beta, state, mask = inputs
    primals = [q, k, v, g, beta, state]
    argnums = (0, 1, 2, 3, 4, 5)
    reference_fn = partial(
        _scalar_recurrence, gated_delta.gated_delta_ops, mask, None
    )
    checkpoint_fn = partial(
        _scalar_recurrence,
        gdn_checkpoint.gated_delta_ops_checkpointed,
        mask,
        chunk,
    )
    expected = mx.value_and_grad(reference_fn, argnums=argnums)(*primals)
    actual = mx.value_and_grad(checkpoint_fn, argnums=argnums)(*primals)
    mx.eval(*expected, *actual)
    np.testing.assert_allclose(
        np.asarray(expected[0]), np.asarray(actual[0]), atol=GRADIENT_ATOL, rtol=0
    )
    for expected_grad, actual_grad in zip(expected[1], actual[1]):
        np.testing.assert_allclose(
            np.asarray(expected_grad),
            np.asarray(actual_grad),
            atol=GRADIENT_ATOL,
            rtol=0,
        )


def _scalar_recurrence(
    recurrence: Callable[..., tuple[Array, Array]],
    mask: Optional[Array],
    chunk: Optional[int],
    q: Array,
    k: Array,
    v: Array,
    g: Array,
    beta: Array,
    state: Array,
) -> Array:
    kwargs = {} if chunk is None else {"chunk": chunk}
    output, final_state = recurrence(q, k, v, g, beta, state, mask, **kwargs)
    return output.sum() + final_state.sum()


@pytest.mark.parametrize("timesteps", [1, 63, 64, 65, 200])
@pytest.mark.parametrize("chunk", [16, 64])
@pytest.mark.parametrize("masked", [False, True])
def test_forward_and_gradients_match_ops(
    timesteps: int, chunk: int, masked: bool
) -> None:
    inputs = _make_inputs(timesteps, masked)
    _assert_forward_matches(inputs, chunk)
    _assert_gradients_match(inputs, chunk)


def _make_tiny_moe(
    num_lora_layers: int = 4, full_attention_interval: int = 2
) -> nn.Module:
    text_config = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "vocab_size": 1000,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-6,
        "linear_num_value_heads": 4,
        "linear_num_key_heads": 2,
        "linear_key_head_dim": 8,
        "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": full_attention_interval,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
        "attention_bias": False,
        "rope_parameters": {
            "type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    }
    args = ModelArgs.from_dict(
        {"model_type": "qwen3_5_moe", "text_config": text_config}
    )
    model = Model(args)
    model.freeze()
    linear_to_lora_layers(
        model,
        num_lora_layers,
        {"rank": 2, "scale": 2.0, "dropout": 0.0},
    )
    model.train()
    return model


def _model_loss(model: nn.Module, inputs: Array, targets: Array) -> Array:
    return nn.losses.cross_entropy(model(inputs), targets).mean()


def _assert_tree_close(expected: object, actual: object) -> None:
    expected_items = tree_flatten(expected)
    actual_items = tree_flatten(actual)
    assert [key for key, _ in expected_items] == [key for key, _ in actual_items]
    for (_, expected_value), (_, actual_value) in zip(expected_items, actual_items):
        np.testing.assert_allclose(
            np.asarray(expected_value),
            np.asarray(actual_value),
            atol=GRADIENT_ATOL,
            rtol=0,
        )


def test_install_matches_tiny_moe_loss_and_lora_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx.random.seed(321)
    baseline_update = (
        gdn_checkpoint._ORIGINAL_UPDATE or qwen3_5.gated_delta_update
    )
    monkeypatch.setattr(qwen3_5, "gated_delta_update", baseline_update)
    model = _make_tiny_moe()
    inputs = mx.array([[17, 42, 81, 109, 211]], dtype=mx.int32)
    targets = mx.array([[42, 81, 109, 211, 307]], dtype=mx.int32)
    value_grad = nn.value_and_grad(model, _model_loss)
    expected_loss, expected_grads = value_grad(model, inputs, targets)
    mx.eval(expected_loss, expected_grads)

    gdn_checkpoint.install()
    actual_loss, actual_grads = value_grad(model, inputs, targets)
    mx.eval(actual_loss, actual_grads)
    np.testing.assert_allclose(
        np.asarray(expected_loss),
        np.asarray(actual_loss),
        atol=GRADIENT_ATOL,
        rtol=0,
    )
    _assert_tree_close(expected_grads, actual_grads)


def test_install_delegates_use_kernel_true_to_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_update = qwen3_5.gated_delta_update
    original_update = gdn_checkpoint._ORIGINAL_UPDATE or previous_update
    monkeypatch.setattr(gdn_checkpoint, "_ORIGINAL_UPDATE", original_update)
    calls = []
    sentinel = object()

    def spy(*args: object, **kwargs: object) -> object:
        calls.append(kwargs.get("use_kernel"))
        return sentinel

    monkeypatch.setattr(qwen3_5, "gated_delta_update", spy)
    gdn_checkpoint.install()
    values = [mx.array(0.0) for _ in range(7)]
    result = qwen3_5.gated_delta_update(*values, use_kernel=True)
    assert result is sentinel
    assert calls == [True]
