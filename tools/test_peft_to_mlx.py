"""Fixture test for tools/peft_to_mlx.py (brief 51): a PEFT LoRA converted for mlx-lm must give, through
mlx-lm's own load_adapters on a tiny quantized qwen3_5 model, exactly W x + scale * B A x computed in numpy.

  .venv/bin/python -m pytest -q tools/test_peft_to_mlx.py
"""
import json
import math
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5
from mlx_lm.tuner.utils import load_adapters
from safetensors import safe_open
from safetensors.numpy import save_file

sys.path.insert(0, str(Path(__file__).parent))
from peft_to_mlx import convert  # noqa: E402

BASE = "Qwen/Qwen3.8-27B"
RANK = 4
ALPHA = 12
GROUP_SIZE = 32
BITS = 4
NUM_LAYERS = 4
TARGETS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
TEXT_CONFIG = dict(model_type="qwen3_5_text", hidden_size=64, intermediate_size=96, num_hidden_layers=NUM_LAYERS,
                   num_attention_heads=2, num_key_value_heads=1, head_dim=32, vocab_size=128,
                   linear_num_value_heads=2, linear_num_key_heads=1, linear_key_head_dim=32,
                   linear_value_head_dim=32, full_attention_interval=4, rms_norm_eps=1e-6, tie_word_embeddings=False)
PEFT_CONFIG = dict(peft_type="LORA", base_model_name_or_path=BASE, r=RANK, lora_alpha=ALPHA, lora_dropout=0.0,
                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                   bias="none", lora_bias=False, use_dora=False, use_rslora=False, fan_in_fan_out=False,
                   modules_to_save=None, rank_pattern={}, alpha_pattern={}, layers_to_transform=None)
PEFT_PREFIX = "base_model.model.model.language_model.layers"
MLX_PREFIX = "language_model.model.layers"


def build_model() -> nn.Module:
    model = qwen3_5.Model(qwen3_5.ModelArgs(model_type="qwen3_5", text_config=TEXT_CONFIG))
    nn.quantize(model, group_size=GROUP_SIZE, bits=BITS)
    return model


def lora_targets(model: nn.Module) -> dict[str, tuple[int, int]]:
    """'<layer>.<path>' -> (in, out) for every target linear that exists in the model."""
    found = {}
    for i, layer in enumerate(model.language_model.model.layers):
        modules = dict(layer.named_modules())
        for path in TARGETS:
            if path in modules:
                lin = modules[path]
                found[f"{i}.{path}"] = (lin.scales.shape[1] * GROUP_SIZE, lin.weight.shape[0])
    return found


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    mx.random.seed(0)
    model = build_model()
    out = tmp_path / "model"
    out.mkdir()
    mx.save_safetensors(str(out / "model.safetensors"), dict(tree_flatten(model.parameters())))
    config = {"model_type": "qwen3_5", "text_config": TEXT_CONFIG,
              "quantization": {"group_size": GROUP_SIZE, "bits": BITS}}
    (out / "config.json").write_text(json.dumps(config))
    return out


def write_peft(tmp_path: Path, model_dir: Path, config_overrides: dict | None = None,
               extra: dict | None = None, drop: str | None = None) -> Path:
    rng = np.random.default_rng(1)
    targets = lora_targets(load_model(model_dir))
    tensors = {}
    for name, (n_in, n_out) in targets.items():
        layer, path = name.split(".", 1)
        key = f"{PEFT_PREFIX}.{layer}.{path}"
        tensors[f"{key}.lora_A.weight"] = rng.standard_normal((RANK, n_in)).astype(np.float32) * 0.1
        tensors[f"{key}.lora_B.weight"] = rng.standard_normal((n_out, RANK)).astype(np.float32) * 0.1  # B != 0
    tensors.update(extra or {})
    if drop:
        del tensors[drop]
    out = tmp_path / "peft"
    out.mkdir()
    save_file(tensors, str(out / "adapter_model.safetensors"))
    (out / "adapter_config.json").write_text(json.dumps({**PEFT_CONFIG, **(config_overrides or {})}))
    return out


def load_model(model_dir: Path) -> nn.Module:
    model = build_model()
    model.load_weights(str(model_dir / "model.safetensors"))
    return model


def test_converted_adapter_matches_numpy_through_load_adapters(tmp_path: Path, model_dir: Path) -> None:
    peft_dir = write_peft(tmp_path, model_dir)
    out_dir = tmp_path / "mlx"
    convert(peft_dir, model_dir, out_dir, BASE)

    model = load_adapters(load_model(model_dir), str(out_dir))
    peft = safe_open(str(peft_dir / "adapter_model.safetensors"), "numpy")
    scale = ALPHA / RANK
    targets = lora_targets(load_model(model_dir))
    assert len(targets) == NUM_LAYERS * 3 + 4  # 3 mlp per layer, 4 self_attn in the one full-attention layer

    # every LoRA module mlx-lm created is filled by the file (load_weights(strict=False) would hide a gap)
    lora_params = {k for k, _ in tree_flatten(model.parameters()) if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")}
    assert lora_params == set(mx.load(str(out_dir / "adapters.safetensors")))
    assert len(lora_params) == 2 * len(targets)

    rng = np.random.default_rng(2)
    for name, (n_in, _) in targets.items():
        layer, path = name.split(".", 1)
        module = dict(model.language_model.model.layers[int(layer)].named_modules())[path]
        base = module.linear
        w = np.array(mx.dequantize(base.weight, base.scales, base.biases, GROUP_SIZE, BITS))
        a = peft.get_tensor(f"{PEFT_PREFIX}.{layer}.{path}.lora_A.weight")
        b = peft.get_tensor(f"{PEFT_PREFIX}.{layer}.{path}.lora_B.weight")
        x = rng.standard_normal((5, n_in)).astype(np.float32)
        want = x @ w.T + scale * (x @ a.T) @ b.T
        got = np.array(module(mx.array(x)))
        rel = np.linalg.norm(got - want) / np.linalg.norm(want)
        assert rel < 1e-5, f"{name}: rel err {rel}"


def test_output_layout_and_config(tmp_path: Path, model_dir: Path) -> None:
    out_dir = tmp_path / "mlx"
    convert(write_peft(tmp_path, model_dir), model_dir, out_dir, BASE)
    cfg = json.loads((out_dir / "adapter_config.json").read_text())
    assert cfg["fine_tune_type"] == "lora"
    assert cfg["num_layers"] == NUM_LAYERS
    params = cfg["lora_parameters"]
    assert params["rank"] == RANK and params["dropout"] == 0.0
    assert params["scale"] == pytest.approx(ALPHA / RANK)
    assert sorted(params["keys"]) == sorted(TARGETS)
    weights = mx.load(str(out_dir / "adapters.safetensors"))
    a = weights[f"{MLX_PREFIX}.0.mlp.down_proj.lora_a"]
    b = weights[f"{MLX_PREFIX}.0.mlp.down_proj.lora_b"]
    assert a.shape == (96, RANK) and b.shape == (RANK, 64)  # (in, r) and (r, out)
    assert a.dtype == mx.float32


def test_rslora_scale(tmp_path: Path, model_dir: Path) -> None:
    out_dir = tmp_path / "mlx"
    convert(write_peft(tmp_path, model_dir, {"use_rslora": True}), model_dir, out_dir, BASE)
    cfg = json.loads((out_dir / "adapter_config.json").read_text())
    assert cfg["lora_parameters"]["scale"] == pytest.approx(ALPHA / math.sqrt(RANK))


BAD_CONFIGS = [
    {"use_dora": True}, {"peft_type": "LOHA"}, {"bias": "all"}, {"lora_bias": True}, {"fan_in_fan_out": True},
    {"modules_to_save": ["lm_head"]}, {"rank_pattern": {"q_proj": 8}}, {"alpha_pattern": {"q_proj": 8}},
    {"layers_to_transform": [0]}, {"base_model_name_or_path": "Qwen/Qwen3.5-27B"}, {"r": RANK + 1},
]


@pytest.mark.parametrize("override", BAD_CONFIGS, ids=lambda o: next(iter(o)))
def test_refuses_unsupported_config(tmp_path: Path, model_dir: Path, override: dict) -> None:
    out_dir = tmp_path / "mlx"
    with pytest.raises(ValueError):
        convert(write_peft(tmp_path, model_dir, override), model_dir, out_dir, BASE)
    assert not (out_dir / "adapters.safetensors").exists()


def test_refuses_key_with_no_model_layer(tmp_path: Path, model_dir: Path) -> None:
    fake = {f"{PEFT_PREFIX}.0.mlp.fake_proj.lora_A.weight": np.zeros((RANK, 64), np.float32),
            f"{PEFT_PREFIX}.0.mlp.fake_proj.lora_B.weight": np.ones((64, RANK), np.float32)}
    out_dir = tmp_path / "mlx"
    with pytest.raises(ValueError, match="fake_proj"):
        convert(write_peft(tmp_path, model_dir, extra=fake), model_dir, out_dir, BASE)
    assert not (out_dir / "adapters.safetensors").exists()


def test_refuses_lora_a_without_lora_b(tmp_path: Path, model_dir: Path) -> None:
    peft_dir = write_peft(tmp_path, model_dir, drop=f"{PEFT_PREFIX}.1.mlp.up_proj.lora_B.weight")
    with pytest.raises(ValueError, match="up_proj"):
        convert(peft_dir, model_dir, tmp_path / "mlx", BASE)


def test_refuses_shape_that_does_not_fit_the_layer(tmp_path: Path, model_dir: Path) -> None:
    key = f"{PEFT_PREFIX}.0.mlp.up_proj.lora_A.weight"
    wrong = {key: np.ones((RANK, 96), np.float32)}  # up_proj takes 64 inputs, not 96
    with pytest.raises(ValueError, match="up_proj"):
        convert(write_peft(tmp_path, model_dir, extra=wrong), model_dir, tmp_path / "mlx", BASE)
