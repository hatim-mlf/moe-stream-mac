"""Parity and bounded-projection tests for the Qwen3.5 MoE loss."""

import numpy as np
import pytest
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5_moe import Model, ModelArgs
from mlx_lm.tuner.trainer import default_loss
from mlx_lm.tuner.utils import linear_to_lora_layers

import ft_lowmem_loss


def _make_model() -> nn.Module:
    text_config = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
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
        "full_attention_interval": 2,
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
        model, 2, {"rank": 2, "scale": 2.0, "dropout": 0.0}
    )
    model.train()
    return model


def _make_batch(
    prompt_completion: list[tuple[int, int]],
) -> tuple[mx.array, mx.array]:
    spans = [(prompt, prompt + completion) for prompt, completion in prompt_completion]
    width = max(end for _, end in spans) + 1
    tokens = np.random.randint(0, 1000, (len(spans), width), dtype=np.int32)
    return mx.array(tokens), mx.array(spans, dtype=mx.int32)


def _assert_gradient_close(expected: object, actual: object) -> None:
    expected_items = tree_flatten(expected)
    actual_items = tree_flatten(actual)
    assert [key for key, _ in expected_items] == [key for key, _ in actual_items]
    for (_, expected_value), (_, actual_value) in zip(expected_items, actual_items):
        np.testing.assert_allclose(
            np.asarray(expected_value), np.asarray(actual_value), atol=1e-4, rtol=1e-4
        )


@pytest.mark.parametrize("chunk_size", [3])
@pytest.mark.parametrize(
    "prompt_completion",
    [
        [(0, 6)],
        [(4, 11)],
        [(0, 5), (3, 10)],
    ],
)
def test_loss_and_trainable_gradients_match_default(
    monkeypatch: pytest.MonkeyPatch,
    chunk_size: int,
    prompt_completion: list[tuple[int, int]],
) -> None:
    mx.random.seed(123)
    np.random.seed(123)
    monkeypatch.setattr(ft_lowmem_loss, "CHUNK", chunk_size)
    model = _make_model()
    batch, lengths = _make_batch(prompt_completion)

    (expected_ce, expected_ntoks), expected_grad = nn.value_and_grad(
        model, default_loss
    )(
        model, batch, lengths
    )
    (actual_ce, actual_ntoks), actual_grad = nn.value_and_grad(
        model, ft_lowmem_loss.completion_only_loss
    )(model, batch, lengths)
    mx.eval(
        expected_ce,
        expected_ntoks,
        expected_grad,
        actual_ce,
        actual_ntoks,
        actual_grad,
    )

    assert abs(expected_ce.item() - actual_ce.item()) <= 1e-4
    assert expected_ntoks.item() == actual_ntoks.item()
    _assert_gradient_close(expected_grad, actual_grad)


class HeadSpy(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.position_counts: list[int] = []

    def __call__(self, hidden: mx.array) -> mx.array:
        self.position_counts.append(hidden.shape[-2])
        return self.inner(hidden)


def test_head_projection_never_exceeds_chunk(monkeypatch: pytest.MonkeyPatch) -> None:
    mx.random.seed(456)
    np.random.seed(456)
    monkeypatch.setattr(ft_lowmem_loss, "CHUNK", 3)
    model = _make_model()
    original_head = model.language_model.lm_head
    spy = HeadSpy(original_head)
    model.language_model.lm_head = spy
    batch, lengths = _make_batch([(2, 13)])

    loss, ntoks = ft_lowmem_loss.completion_only_loss(model, batch, lengths)
    mx.eval(loss, ntoks)

    assert spy.position_counts
    assert max(spy.position_counts) <= ft_lowmem_loss.CHUNK
    assert sum(spy.position_counts) == ntoks.item()
