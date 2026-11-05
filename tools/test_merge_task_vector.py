#!/usr/bin/env python3
"""Synthetic end-to-end tests for the NumPy MLX task-vector merger."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file, save_file

import merge_task_vector as merge


GROUP = 64


def _arrays(seed: int, scale: float = 1.0) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    quantized = {
        "language_model.model.layers.0.self_attn.q_proj.weight": rng.normal(0, scale, (4, 128)).astype(np.float32),
        "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight": rng.normal(
            0, scale, (256, 4, 128)
        ).astype(np.float32),
        "language_model.model.layers.0.mlp.shared_expert.up_proj.weight": rng.normal(
            0, scale, (4, 128)
        ).astype(np.float32),
        "language_model.model.layers.0.mlp.gate.weight": rng.normal(0, scale, (4, 128)).astype(np.float32),
    }
    dense = {
        "language_model.model.layers.0.linear_attn.A_log": rng.normal(0, scale, (4,)).astype(np.float32),
    }
    return quantized, dense


def _write_checkpoint(
    directory: Path,
    quantized: dict[str, np.ndarray],
    dense: dict[str, np.ndarray],
    qbits: dict[str, int],
    tokenizer: bytes = b'{"model":{"type":"WordLevel","vocab":{"hello":0,"</s>":1}},"added_tokens":[]}',
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, np.ndarray] = {}
    overrides: dict[str, dict[str, int]] = {}
    for weight_key, values in quantized.items():
        bits = qbits[weight_key]
        packed, scales, biases = merge.quantize_affine(values, bits, GROUP)
        stem = weight_key[: -len(".weight")]
        tensors[weight_key] = packed
        tensors[stem + ".scales"] = scales.astype(np.float32)
        tensors[stem + ".biases"] = biases.astype(np.float32)
        if bits != 4:
            overrides[stem] = {"bits": bits, "group_size": GROUP}
    tensors.update(dense)
    shard_name = "model-00001-of-00001.safetensors"
    save_file(tensors, str(directory / shard_name))
    weight_map = {key: shard_name for key in tensors}
    size = sum(int(value.nbytes) for value in tensors.values())
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": size}, "weight_map": weight_map}, indent=2) + "\n"
    )
    config = {
        "model_type": "qwen3_5_moe",
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 1,
            "num_experts": 256,
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 128,
        },
        "quantization": {"bits": 4, "group_size": GROUP, "mode": "affine", **overrides},
    }
    (directory / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (directory / "tokenizer.json").write_bytes(tokenizer)
    (directory / "tokenizer_config.json").write_text(
        '{"synthetic":true,"eos_token":"</s>","eos_token_id":1}\n'
    )
    (directory / "chat_template.jinja").write_text("synthetic template\n")


def _read_triplet(checkpoint_dir: Path, weight_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensors = load_file(str(checkpoint_dir / "model-00001-of-00001.safetensors"))
    stem = weight_key[: -len(".weight")]
    return tensors[weight_key], tensors[stem + ".scales"], tensors[stem + ".biases"]


class MergeTaskVectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="merge-task-vector-")
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.ft = self.root / "finetune"
        base_q, base_f = _arrays(10, 0.7)
        ft_q, ft_f = _arrays(20, 0.8)
        base_bits = {
            key: (8 if "shared_expert" in key or "mlp.gate." in key else 4) for key in base_q
        }
        ft_bits = {
            key: (4 if "shared_expert" in key or "mlp.gate." in key else 8) for key in ft_q
        }
        self.base_quantized = base_q
        self.ft_quantized = ft_q
        self.base_dense = base_f
        self.ft_dense = ft_f
        self.base_bits = base_bits
        _write_checkpoint(self.base, base_q, base_f, base_bits)
        _write_checkpoint(self.ft, ft_q, ft_f, ft_bits)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_lam_zero_preserves_all_base_tensor_bytes(self) -> None:
        out = self.root / "zero"
        merge.merge_checkpoints(self.base, self.ft, out, lam=0.0, verbose=False)
        base_tensors = load_file(str(self.base / "model-00001-of-00001.safetensors"))
        out_tensors = load_file(str(out / "model-00001-of-00001.safetensors"))
        self.assertEqual(set(base_tensors), set(out_tensors))
        for key in base_tensors:
            self.assertEqual(base_tensors[key].tobytes(), out_tensors[key].tobytes(), key)

    def test_lam_one_is_finetune_dequantized_then_base_requantized(self) -> None:
        out = self.root / "one"
        merge.merge_checkpoints(self.base, self.ft, out, lam=1.0, verbose=False)
        result = load_file(str(out / "model-00001-of-00001.safetensors"))
        for weight_key, base_values in self.base_quantized.items():
            ft_q, ft_s, ft_b = _read_triplet(self.ft, weight_key)
            ft_bits = 8 if "shared_expert" not in weight_key and "mlp.gate." not in weight_key else 4
            ft_values = merge.dequantize_affine(ft_q, ft_s, ft_b, ft_bits, GROUP)
            expected_q, expected_s, expected_b = merge.quantize_affine(
                ft_values, self.base_bits[weight_key], GROUP
            )
            stem = weight_key[: -len(".weight")]
            np.testing.assert_array_equal(result[weight_key], expected_q)
            np.testing.assert_array_equal(result[stem + ".scales"], expected_s)
            np.testing.assert_array_equal(result[stem + ".biases"], expected_b)

    def test_quantize_dequantize_error_is_bounded_by_one_scale(self) -> None:
        rng = np.random.default_rng(123)
        values = rng.normal(size=(7, 128)).astype(np.float32)
        for bits in (4, 8):
            packed, scales, biases = merge.quantize_affine(values, bits, GROUP)
            reconstructed = merge.dequantize_affine(packed, scales, biases, bits, GROUP)
            scale_per_value = np.repeat(np.abs(scales), GROUP, axis=-1)
            error = np.abs(values - reconstructed)
            # MLX's endpoint correction can make the representable range
            # slightly narrower than the input range, so clipped endpoints
            # can exceed the usual half-step rounding bound by up to one step.
            np.testing.assert_array_less(error, scale_per_value + 2e-6)

    def test_mlx_affine_endpoint_probes(self) -> None:
        left_edge = np.tile(np.array([-1.0, -0.2, 0.3, 0.5], dtype=np.float32), 16)
        right_edge = np.tile(np.array([-0.2, 0.1, 0.3, 1.0], dtype=np.float32), 16)
        _, scales_left, biases_left = merge.quantize_affine(left_edge, 4, GROUP)
        _, scales_right, biases_right = merge.quantize_affine(right_edge, 4, GROUP)
        np.testing.assert_array_equal(scales_left, np.array([0.1], dtype=np.float32))
        np.testing.assert_array_equal(biases_left, np.array([-1.0], dtype=np.float32))
        np.testing.assert_array_equal(scales_right, np.array([-0.083333336], dtype=np.float32))
        np.testing.assert_array_equal(biases_right, np.array([1.0], dtype=np.float32))

    def test_semantic_tokenizer_check_ignores_json_formatting(self) -> None:
        self.ft.joinpath("tokenizer.json").write_text(
            '{ "model" : { "vocab" : { "hello" : 0, "</s>" : 1 }, "type" : "WordLevel" }, '
            '"added_tokens" : [ ] }\n'
        )
        base = merge.Checkpoint(self.base)
        finetune = merge.Checkpoint(self.ft)
        try:
            base_hash, ft_hash = merge._tokenizer_hash(base), merge._tokenizer_hash(finetune)
            self.assertNotEqual(base_hash, ft_hash)
            self.assertEqual(
                merge.compatibility_issues(base, finetune, base_hash, ft_hash), []
            )
        finally:
            base.close()
            finetune.close()

    def test_text_only_copies_base_only_tensors(self) -> None:
        base_q, base_dense = _arrays(50, 0.7)
        ft_q, ft_dense = _arrays(50, 0.7)
        bits = {key: (8 if "shared_expert" in key or "mlp.gate." in key else 4) for key in base_q}
        base_dense["vision_tower.patch_embed.proj.weight"] = np.arange(128, dtype=np.float32)
        base_dir, ft_dir, out = self.root / "text-base", self.root / "text-ft", self.root / "text-out"
        _write_checkpoint(base_dir, base_q, base_dense, bits)
        _write_checkpoint(ft_dir, ft_q, ft_dense, bits)
        meta = merge.merge_checkpoints(
            base_dir, ft_dir, out, lam=0.5, text_only=True, verbose=False
        )
        base_tensors = load_file(str(base_dir / "model-00001-of-00001.safetensors"))
        out_tensors = load_file(str(out / "model-00001-of-00001.safetensors"))
        vision_key = "vision_tower.patch_embed.proj.weight"
        self.assertEqual(base_tensors[vision_key].tobytes(), out_tensors[vision_key].tobytes())
        self.assertEqual(meta["copied_tensors_count"], 1)
        self.assertEqual(meta["copied_keys_count"], 1)

    def test_text_only_still_rejects_language_model_key_mismatch(self) -> None:
        bad_dense = dict(self.ft_dense)
        bad_dense.pop("language_model.model.layers.0.linear_attn.A_log")
        bad = self.root / "text-bad"
        _write_checkpoint(bad, self.ft_quantized, bad_dense, self.base_bits)
        with self.assertRaisesRegex(ValueError, "logical key-set mismatch"):
            merge.merge_checkpoints(
                self.base, bad, self.root / "text-bad-out", text_only=True, verbose=False
            )

    def test_dare_p_zero_equals_linear(self) -> None:
        linear_out, dare_out = self.root / "linear", self.root / "dare-zero"
        merge.merge_checkpoints(self.base, self.ft, linear_out, lam=0.6, method="linear", verbose=False)
        merge.merge_checkpoints(
            self.base, self.ft, dare_out, lam=0.6, method="dare", dare_p=0.0, seed=7, verbose=False
        )
        linear = load_file(str(linear_out / "model-00001-of-00001.safetensors"))
        dare = load_file(str(dare_out / "model-00001-of-00001.safetensors"))
        for key in linear:
            self.assertEqual(linear[key].tobytes(), dare[key].tobytes(), key)

    def test_ties_trim_keeps_top_magnitudes_without_rescaling(self) -> None:
        q_base, _ = _arrays(40)
        q_ft, _ = _arrays(41)
        dense_key = "language_model.model.layers.0.linear_attn.A_log"
        dense_base = {dense_key: np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)}
        dense_ft = {dense_key: np.array([10.0, 2.0, -7.0, 9.0], dtype=np.float32)}
        base_dir, ft_dir, out = self.root / "ties-base", self.root / "ties-ft", self.root / "ties-out"
        _write_checkpoint(base_dir, q_base, dense_base, self.base_bits)
        _write_checkpoint(ft_dir, q_ft, dense_ft, {
            key: (4 if "shared_expert" in key or "mlp.gate." in key else 8) for key in q_ft
        })
        merge.merge_checkpoints(
            base_dir, ft_dir, out, lam=0.5, method="ties", ties_trim=0.5, verbose=False
        )
        result = load_file(str(out / "model-00001-of-00001.safetensors"))
        np.testing.assert_array_equal(result[dense_key], np.array([5.0, 1.0, -2.5, 3.0], dtype=np.float32))

    def test_ties_keep_fraction_one_matches_linear(self) -> None:
        linear_out, ties_out = self.root / "linear-all", self.root / "ties-all"
        merge.merge_checkpoints(self.base, self.ft, linear_out, lam=0.4, method="linear", verbose=False)
        merge.merge_checkpoints(
            self.base, self.ft, ties_out, lam=0.4, method="ties", ties_trim=1.0, verbose=False
        )
        linear = load_file(str(linear_out / "model-00001-of-00001.safetensors"))
        ties = load_file(str(ties_out / "model-00001-of-00001.safetensors"))
        for key in linear:
            self.assertEqual(linear[key].tobytes(), ties[key].tobytes(), key)

    def test_mismatched_logical_shapes_and_keys_raise(self) -> None:
        q, dense = _arrays(30)
        shape_bad = self.root / "shape-bad"
        dense["language_model.model.layers.0.linear_attn.A_log"] = np.ones((5,), dtype=np.float32)
        _write_checkpoint(shape_bad, q, dense, self.base_bits)
        with self.assertRaisesRegex(ValueError, "shape mismatch"):
            merge.merge_checkpoints(self.base, shape_bad, self.root / "bad-shape", verbose=False)

        q, dense = _arrays(30)
        dense.clear()
        key_bad = self.root / "key-bad"
        _write_checkpoint(key_bad, q, dense, self.base_bits)
        with self.assertRaisesRegex(ValueError, "key-set mismatch"):
            merge.merge_checkpoints(self.base, key_bad, self.root / "bad-keys", verbose=False)

    def test_per_module_lambda_overrides_only_matching_paths(self) -> None:
        out = self.root / "overrides"
        overrides = {"attn": 0.5, "experts": 0.6, "shared": 0.2, "router": 0.3}
        merge.merge_checkpoints(
            self.base,
            self.ft,
            out,
            lam=0.1,
            lam_attn=overrides["attn"],
            lam_experts=overrides["experts"],
            lam_shared=overrides["shared"],
            lam_router=overrides["router"],
            verbose=False,
        )
        actual = load_file(str(out / "model-00001-of-00001.safetensors"))
        for key, base_values in self.base_quantized.items():
            module = merge.module_for_tensor(key)
            lam = overrides.get(module, 0.1)
            base_q, base_s, base_b = _read_triplet(self.base, key)
            ft_q, ft_s, ft_b = _read_triplet(self.ft, key)
            base_float = merge.dequantize_affine(base_q, base_s, base_b, self.base_bits[key], GROUP)
            ft_bits = 8 if "shared_expert" not in key and "mlp.gate." not in key else 4
            ft_float = merge.dequantize_affine(ft_q, ft_s, ft_b, ft_bits, GROUP)
            expected = merge.quantize_affine(base_float + lam * (ft_float - base_float), self.base_bits[key], GROUP)
            stem = key[: -len(".weight")]
            for out_key, expected_array in zip((key, stem + ".scales", stem + ".biases"), expected):
                np.testing.assert_array_equal(actual[out_key], expected_array, err_msg=key)

        dense_key = "language_model.model.layers.0.linear_attn.A_log"
        dense_actual = actual[dense_key]
        expected_dense = self.base_dense[dense_key] + overrides["attn"] * (
            self.ft_dense[dense_key] - self.base_dense[dense_key]
        )
        np.testing.assert_array_equal(dense_actual, expected_dense)


if __name__ == "__main__":
    unittest.main(verbosity=2)
