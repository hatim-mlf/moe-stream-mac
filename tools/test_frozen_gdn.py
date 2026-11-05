"""Parity and dispatch tests for frozen lower Qwen3.5 GDN layers."""

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten

import frozen_gdn
from test_gdn_checkpoint import _make_tiny_moe


INPUT_IDS = [[17, 42, 81, 109, 211]]
TARGET_IDS = [[42, 81, 109, 211, 307]]
PARITY_ATOL = 1e-3
SEED = 321


def _make_last_layer_lora_model() -> nn.Module:
    mx.random.seed(SEED)
    return _make_tiny_moe(num_lora_layers=1, full_attention_interval=3)


def _model_loss(model: nn.Module, inputs: mx.array, targets: mx.array) -> mx.array:
    return nn.losses.cross_entropy(model(inputs), targets).mean()


def _loss_and_gradients(model: nn.Module) -> tuple[np.ndarray, list[tuple[str, Any]]]:
    inputs = mx.array(INPUT_IDS, dtype=mx.int32)
    targets = mx.array(TARGET_IDS, dtype=mx.int32)
    loss, gradients = nn.value_and_grad(model, _model_loss)(model, inputs, targets)
    mx.eval(loss, gradients)
    return np.asarray(loss), tree_flatten(gradients)


def _assert_parity(
    expected: tuple[np.ndarray, list[tuple[str, Any]]],
    actual: tuple[np.ndarray, list[tuple[str, Any]]],
) -> None:
    expected_loss, expected_gradients = expected
    actual_loss, actual_gradients = actual
    assert [key for key, _ in expected_gradients] == [
        key for key, _ in actual_gradients
    ]
    differences = [float(np.max(np.abs(expected_loss - actual_loss)))]
    for (_, expected_grad), (_, actual_grad) in zip(
        expected_gradients, actual_gradients
    ):
        expected_array = np.asarray(expected_grad)
        actual_array = np.asarray(actual_grad)
        differences.append(float(np.max(np.abs(expected_array - actual_array))))
    max_diff = max(differences, default=0.0)
    print(f"frozen_gdn_max_abs_loss_or_gradient_diff={max_diff:.7g}")
    np.testing.assert_allclose(expected_loss, actual_loss, atol=PARITY_ATOL, rtol=0)
    for (_, expected_grad), (_, actual_grad) in zip(
        expected_gradients, actual_gradients
    ):
        np.testing.assert_allclose(
            np.asarray(expected_grad), np.asarray(actual_grad), atol=PARITY_ATOL, rtol=0
        )


def test_install_matches_ops_loss_and_last_layer_lora_gradients() -> None:
    baseline = _make_last_layer_lora_model()
    expected = _loss_and_gradients(baseline)

    optimized = _make_last_layer_lora_model()
    frozen_gdn.install(optimized)
    actual = _loss_and_gradients(optimized)

    _assert_parity(expected, actual)


def test_frozen_and_trainable_gdn_dispatch_survives_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_last_layer_lora_model()
    frozen_gdn.install(model)
    original_update = frozen_gdn._ORIGINAL_UPDATE
    assert original_update is not None
    calls: list[bool] = []

    def spy(*args: Any, **kwargs: Any) -> Any:
        calls.append(bool(kwargs.get("use_kernel")))
        return original_update(*args, **kwargs)

    monkeypatch.setattr(frozen_gdn, "_ORIGINAL_UPDATE", spy)
    inputs = mx.array(INPUT_IDS, dtype=mx.int32)
    mx.eval(model(inputs))
    assert calls == [True, True, False]

    calls.clear()
    model.train()
    mx.eval(model(inputs))
    assert calls == [True, True, False]


def test_install_requires_trainable_parameters() -> None:
    model = _make_last_layer_lora_model()
    model.freeze()
    with pytest.raises(ValueError, match="trainable parameter"):
        frozen_gdn.install(model)
