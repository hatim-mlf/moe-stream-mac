#!/usr/bin/env python3
"""Unit tests for residual steering; numpy checks run without an MLX device."""
import sys
import unittest

import numpy as np

import steering


try:
    import mlx.core as mx
except Exception as error:  # Metal is unavailable in headless CI/sandbox runs.
    mx = None
    qwen3_5 = None
    _MLX_ERROR = error
else:
    from mlx_lm.models import qwen3_5
    _MLX_ERROR = None


class FakeLayer:
    def __init__(self, hidden_size=3):
        self.input_layernorm = type("Norm", (), {"weight": np.ones(hidden_size)})()
        self.is_linear = False

    def __call__(self, x, **kwargs):
        return x + 1


class FakeInner:
    def __init__(self):
        self.layers = [FakeLayer()]


class FakeModel:
    def __init__(self):
        self.model = FakeInner()


def tiny_model():
    config = {
        "model_type": "qwen3_5_moe",
        "hidden_size": 16,
        "intermediate_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": 32,
        "max_position_embeddings": 64,
        "full_attention_interval": 1,
        "partial_rotary_factor": 1.0,
        "rope_parameters": {"type": "default", "rope_theta": 10000},
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 8,
        "shared_expert_intermediate_size": 8,
        "norm_topk_prob": True,
        "tie_word_embeddings": False,
    }
    return qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5_moe", text_config=config))


class TinyTokenizer:
    def _content_ids(self, text):
        return [5 + (ord(char) % 25) for char in text]

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True,
                            enable_thinking=False):
        ids = [1]
        for message in messages:
            role = message["role"]
            if role == "assistant":
                ids.append(4)
            else:
                ids.append(3 if role == "system" else 6)
            ids.extend(self._content_ids(message["content"]))
            ids.append(2)
        if add_generation_prompt and (not messages or messages[-1]["role"] != "assistant"):
            ids.append(4)
        return ids


class NumpyTests(unittest.TestCase):
    def test_random_directions_are_seeded_and_have_requested_norm(self):
        left = steering.random_directions(19, 5, 42, 3.25)
        right = steering.random_directions(19, 5, 42, 3.25)
        self.assertEqual(left.shape, (5, 19))
        np.testing.assert_array_equal(left, right)
        np.testing.assert_allclose(np.linalg.norm(left, axis=1), 3.25, rtol=1e-6)

    def test_random_zero_norm(self):
        values = steering.random_directions(4, 2, 0, 0)
        np.testing.assert_array_equal(values, np.zeros((2, 4), dtype=np.float32))

    def test_build_directions_and_rms(self):
        preferred = steering.MeanCapture({2: np.array([3, 2], np.float64),
                                          5: np.array([1, 6], np.float64)},
                                         residual_rms={2: 4.0, 5: 2.0})
        unpreferred = steering.MeanCapture({2: np.array([1, 0], np.float64),
                                            5: np.array([0, 2], np.float64)},
                                           residual_rms={2: 2.0, 5: 4.0})
        directions, residual_rms = steering.build_directions(preferred, unpreferred)
        np.testing.assert_array_equal(directions, np.array([[2, 2], [1, 4]], dtype=np.float32))
        np.testing.assert_allclose(residual_rms, [np.sqrt(10), np.sqrt(10)], rtol=1e-6)

    def test_build_directions_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            steering.build_directions({}, {})
        with self.assertRaises(ValueError):
            steering.build_directions({0: np.ones(3)}, {1: np.ones(3)})
        with self.assertRaises(ValueError):
            steering.build_directions({0: np.array([np.nan])}, {0: np.array([0.0])})
        with self.assertRaises(ValueError):
            steering.build_directions({0: np.ones((1, 3))}, {0: np.ones((1, 3))})

    def test_random_directions_reject_invalid_inputs(self):
        for args in ((0, 1, 0, 1), (4, 0, 0, 1), (4, 1, -1, 1),
                     (4, 1, 0, -1), (4, 1, 0, np.inf), (True, 1, 0, 1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                steering.random_directions(*args)

    def test_find_layers_supports_fallback_and_rejects_missing(self):
        model = FakeModel()
        self.assertIs(steering.find_layers(model), model.model.layers)
        with self.assertRaises(ValueError):
            steering.find_layers(object())


@unittest.skipIf(_MLX_ERROR is not None, f"MLX unavailable: {_MLX_ERROR}")
class MlxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        mx.random.seed(20261105)

    def test_zero_alpha_is_bit_identical_and_remove_restores(self):
        model = tiny_model()
        ids = mx.array([[1]])
        baseline = model(ids)
        mx.eval(baseline)
        before = np.asarray(baseline).copy()
        original = steering.find_layers(model)[0]
        steering.install_steering(model, 0, np.ones(16, dtype=np.float32), 0.0)
        steered = model(ids)
        mx.eval(steered)
        np.testing.assert_array_equal(np.asarray(steered), before)
        steering.remove_steering(model)
        self.assertIs(steering.find_layers(model)[0], original)
        restored = model(ids)
        mx.eval(restored)
        np.testing.assert_array_equal(np.asarray(restored), before)

    def test_hook_changes_only_single_token_calls_by_exact_delta(self):
        model = tiny_model()
        layers = steering.find_layers(model)
        original = layers[0]
        vector = np.linspace(-0.4, 0.6, 16, dtype=np.float32)
        alpha = 0.25
        prefill = mx.array(np.arange(32, dtype=np.float32).reshape(1, 2, 16) / 32)
        decode = mx.array(np.arange(16, dtype=np.float32).reshape(1, 1, 16) / 16)
        prefill_base = original(prefill)
        decode_base = original(decode)
        mx.eval(prefill_base, decode_base)
        steering.install_steering(model, 0, vector, alpha)
        wrapped = layers[0]
        prefill_hook = wrapped(prefill)
        decode_hook = wrapped(decode)
        mx.eval(prefill_hook, decode_hook)
        np.testing.assert_array_equal(np.asarray(prefill_hook), np.asarray(prefill_base))
        expected = np.asarray(decode_base) + (alpha * vector).astype(np.float32).reshape(1, 1, -1)
        np.testing.assert_array_equal(np.asarray(decode_hook), expected)
        steering.remove_steering(model)

    def test_capture_means_matches_brute_force_answer_token_means(self):
        model = tiny_model()
        tokenizer = TinyTokenizer()
        messages = [{"role": "system", "content": "rules"},
                    {"role": "user", "content": "return x"}]
        answer = "x=2"
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                               tokenize=True, enable_thinking=False)
        full = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": answer}],
                                             tokenize=True, enable_thinking=False)
        self.assertEqual(full[:len(prompt)], prompt)
        means = steering.capture_means(model, tokenizer,
                                       [{"messages": messages, "answer": answer}], [0, 1])

        language = model.language_model
        inner = language.model
        tokens = mx.array([full])
        hidden = inner.embed_tokens(tokens)
        mask = qwen3_5.create_attention_mask(hidden, None)
        brute = {}
        for layer_id, layer in enumerate(inner.layers):
            hidden = layer(hidden, mask=mask, cache=None)
            mx.eval(hidden)
            brute[layer_id] = np.asarray(hidden[:, len(prompt):len(full), :]).mean(axis=(0, 1))
        for layer_id in (0, 1):
            np.testing.assert_allclose(means[layer_id], brute[layer_id], rtol=1e-5, atol=1e-5)
            self.assertGreater(means.residual_rms[layer_id], 0)

    def test_install_rejects_invalid_values(self):
        model = tiny_model()
        with self.assertRaises(ValueError):
            steering.install_steering(model, -1, np.ones(16), 1)
        with self.assertRaises(ValueError):
            steering.install_steering(model, 0, np.ones(15), 1)
        with self.assertRaises(ValueError):
            steering.install_steering(model, 0, np.ones(16), np.nan)
        with self.assertRaises(ValueError):
            steering.install_steering(model, 0, np.full(16, np.inf), 1)


def main():
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if _MLX_ERROR is not None:
        print(f"MLX model tests skipped: {_MLX_ERROR}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
