"""Convert a PEFT LoRA adapter to an mlx-lm adapter."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


LAYER_PATH = re.compile(r"\.layers\.\d+\.(.+)$")


def _read_model_tensor_names(model_dir: Path) -> dict[str, tuple[int, ...]]:
    """Tensor name -> shape, read from the safetensors headers only (each shard opened once)."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        shards = sorted(set(_load_json(index_path)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    result: dict[str, tuple[int, ...]] = {}
    for shard in shards:
        with safe_open(model_dir / shard, framework="numpy") as f:
            for key in f.keys():
                result[key] = tuple(f.get_slice(key).get_shape())
    return result


def _get_bits(quantization: dict[str, Any] | None, prefix: str) -> int:
    """Quantization bits for a quantized layer; refuse when the model config does not say."""
    if quantization is None:
        raise ValueError(f"{prefix}: quantized layer but config.json has no 'quantization'")
    override = quantization.get(prefix)
    return int(override["bits"]) if isinstance(override, dict) else int(quantization["bits"])


def convert(peft_dir: Path, model_dir: Path, out_dir: Path, expected_base: str) -> dict[str, Any]:
    """Validate everything first, then write out_dir/adapters.safetensors and out_dir/adapter_config.json."""
    peft_config = _load_json(peft_dir / "adapter_config.json")
    model_config = _load_json(model_dir / "config.json")

    # Validate adapter_config.json
    if peft_config.get("peft_type", "") != "LORA":
        raise ValueError("peft_type is not LORA")
    if peft_config.get("base_model_name_or_path", "") != expected_base:
        raise ValueError(f"base_model_name_or_path does not match expected {expected_base}")
    if peft_config.get("use_dora", False):
        raise ValueError("use_dora is true")
    if peft_config.get("bias", "none") != "none":
        raise ValueError("bias is not none")
    if peft_config.get("lora_bias", False):
        raise ValueError("lora_bias is true")
    if peft_config.get("fan_in_fan_out", False):
        raise ValueError("fan_in_fan_out is true")
    if peft_config.get("modules_to_save", {}):
        raise ValueError("modules_to_save is non-empty")
    if peft_config.get("rank_pattern", {}):
        raise ValueError("rank_pattern is non-empty")
    if peft_config.get("alpha_pattern", {}):
        raise ValueError("alpha_pattern is non-empty")
    if peft_config.get("layers_to_transform") is not None:
        raise ValueError("layers_to_transform is not None")

    if "r" not in peft_config or "lora_alpha" not in peft_config:
        raise ValueError("adapter_config.json must give both r and lora_alpha")
    r = int(peft_config["r"])
    lora_alpha = peft_config["lora_alpha"]
    use_rslora = peft_config.get("use_rslora", False)
    scale = lora_alpha / (r ** 0.5 if use_rslora else r)

    text_config = model_config.get("text_config")
    if text_config is not None:
        num_layers = text_config.get("num_hidden_layers", 0)
    else:
        num_layers = model_config.get("num_hidden_layers", 0)

    quantization = model_config.get("quantization")

    with safe_open(peft_dir / "adapter_model.safetensors", framework="numpy") as f:
        adapter_tensors = {key: f.get_tensor(key) for key in f.keys()}
    model_tensor_names = _read_model_tensor_names(model_dir)

    pattern = re.compile(r"^base_model\.model\.(.+)\.lora_(A|B)\.weight$")
    prefix_map: dict[str, dict[str, np.ndarray]] = {}

    for key, tensor in adapter_tensors.items():
        match = pattern.match(key)
        if not match:
            raise ValueError(f"Invalid key format: {key}")
        prefix = match.group(1)
        ab = match.group(2)
        # Normalize prefix: replace leading model.language_model. with language_model.model.
        if prefix.startswith("model.language_model."):
            prefix = "language_model.model." + prefix[len("model.language_model."):]
        if prefix not in prefix_map:
            prefix_map[prefix] = {}
        prefix_map[prefix][ab] = tensor

    keys: list[str] = []
    for prefix, tensors in prefix_map.items():
        if "A" not in tensors or "B" not in tensors:
            raise ValueError(f"Missing lora_A or lora_B for prefix: {prefix}")

        model_key = f"{prefix}.weight"
        if model_key not in model_tensor_names:
            raise ValueError(f"Missing model weight for prefix: {prefix}")

        layer_match = LAYER_PATH.search(prefix)
        if layer_match is None:
            raise ValueError(f"Prefix missing layers.<N>: {prefix}")
        module_path = layer_match.group(1)
        if module_path not in keys:
            keys.append(module_path)

        lora_a = tensors["A"]  # (r, in)
        lora_b = tensors["B"]  # (out, r)

        if lora_a.ndim != 2 or lora_a.shape[0] != r:
            raise ValueError(f"lora_A shape {lora_a.shape} is not (r={r}, in) for prefix: {prefix}")
        if lora_b.ndim != 2 or lora_b.shape[1] != r:
            raise ValueError(f"lora_B shape {lora_b.shape} is not (out, r={r}) for prefix: {prefix}")

        out, w1 = model_tensor_names[model_key]
        if lora_b.shape[0] != out:
            raise ValueError(f"lora_B output shape mismatch for prefix: {prefix}")

        if f"{prefix}.scales" in model_tensor_names:
            bits = _get_bits(quantization, prefix)
            in_dim = w1 * 32 // bits
        else:
            in_dim = w1

        if lora_a.shape[1] != in_dim:
            raise ValueError(f"lora_A input shape mismatch for prefix: {prefix}")

    keys.sort()

    out_dir.mkdir(parents=True, exist_ok=True)

    output_tensors: dict[str, np.ndarray] = {}
    for prefix, tensors in prefix_map.items():
        lora_a = np.ascontiguousarray(tensors["A"].T)
        lora_b = np.ascontiguousarray(tensors["B"].T)
        output_tensors[f"{prefix}.lora_a"] = lora_a
        output_tensors[f"{prefix}.lora_b"] = lora_b

    save_file(output_tensors, out_dir / "adapters.safetensors")

    adapter_config = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {
            "rank": r,
            "scale": scale,
            "dropout": 0.0,
            "keys": keys,
        },
        "source": {
            "peft_dir": str(peft_dir),
            "base_model": expected_base,
        },
    }
    with open(out_dir / "adapter_config.json", "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    return {
        "tensors": len(output_tensors),
        "modules": len(prefix_map),
        "rank": r,
        "scale": scale,
        "num_layers": num_layers,
        "keys": keys,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PEFT LoRA adapter to mlx-lm adapter")
    parser.add_argument("peft_dir", type=Path)
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--base", required=True, type=str)
    args = parser.parse_args()

    summary = convert(args.peft_dir, args.model_dir, args.out_dir, args.base)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
