#!/usr/bin/env python3
"""Capture and apply inference-time residual directions for MLX-LM models."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import mem_guard


class MeanCapture(dict):
    """Layer-indexed residual means with pooled per-layer RMS metadata."""

    def __init__(self, *args, residual_rms=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_rms = {} if residual_rms is None else dict(residual_rms)


def find_layers(model) -> list:
    """Return the decoder layer list from an MLX-LM model wrapper."""
    language_model = getattr(model, "language_model", None)
    candidates = []
    if language_model is not None:
        language_inner = getattr(language_model, "model", None)
        candidates.append(getattr(language_inner, "layers", None))
    model_inner = getattr(model, "model", None)
    candidates.append(getattr(model_inner, "layers", None))
    for layers in candidates:
        if layers is not None and hasattr(layers, "__len__") and hasattr(layers, "__getitem__"):
            if len(layers):
                return layers
    raise ValueError("model has no decoder layers at language_model.model.layers or model.layers")


def _mlx():
    try:
        import mlx.core as mx
    except Exception as error:
        raise RuntimeError("MLX is required for residual capture and steering") from error
    return mx


def _hidden_size(model, layer=None):
    candidates = [
        getattr(getattr(model, "args", None), "hidden_size", None),
        getattr(getattr(getattr(model, "language_model", None), "args", None), "hidden_size", None),
        getattr(getattr(getattr(model, "language_model", None), "model", None), "args", None),
    ]
    if candidates[-1] is not None:
        candidates.append(getattr(candidates[-1], "hidden_size", None))
    if layer is not None:
        candidates.append(getattr(getattr(layer, "input_layernorm", None), "weight", None))
    for candidate in candidates:
        if isinstance(candidate, (int, np.integer)) and int(candidate) > 0:
            return int(candidate)
        shape = getattr(candidate, "shape", None)
        if shape and len(shape) == 1 and int(shape[-1]) > 0:
            return int(shape[-1])
    return None


def _validate_layers(layers, layer_ids):
    if layer_ids is None:
        return list(range(len(layers)))
    ids = list(layer_ids)
    if not ids:
        raise ValueError("at least one layer is required")
    if any(not isinstance(i, (int, np.integer)) or isinstance(i, bool) for i in ids):
        raise ValueError("layer indices must be integers")
    if len(set(int(i) for i in ids)) != len(ids):
        raise ValueError("layer indices must be unique")
    if any(int(i) < 0 or int(i) >= len(layers) for i in ids):
        raise ValueError("layer index out of range")
    return [int(i) for i in ids]


def _get_hidden(output):
    if isinstance(output, (tuple, list)) and output:
        return output[0], "tuple" if isinstance(output, tuple) else "list"
    return output, None


def _replace_hidden(output, hidden, kind):
    if kind == "tuple":
        return (hidden, *output[1:])
    if kind == "list":
        return [hidden, *output[1:]]
    return hidden


class _CaptureLayer:
    def __init__(self, layer, layer_id, start, stop, totals):
        self.layer = layer
        self.layer_id = layer_id
        self.start = start
        self.stop = stop
        self.totals = totals

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        mx = _mlx()
        output = self.layer(*args, **kwargs)
        hidden, kind = _get_hidden(output)
        if not hasattr(hidden, "shape") or len(hidden.shape) != 3:
            raise ValueError("decoder layer output must have shape [batch, sequence, hidden]")
        if self.stop > hidden.shape[-2]:
            raise ValueError("answer token positions exceed decoder layer output length")
        answer = hidden[:, self.start:self.stop, :]
        token_count = int(answer.shape[0] * answer.shape[1])
        if token_count <= 0:
            raise ValueError("assistant answer has no positions in the decoder output")
        summed = mx.sum(answer.astype(mx.float32), axis=(0, 1))
        sum_squares = mx.sum(answer.astype(mx.float32) * answer.astype(mx.float32))
        mx.eval(summed, sum_squares)
        stats = self.totals[self.layer_id]
        layer_sum = np.asarray(summed, dtype=np.float64)
        stats["sum"] = layer_sum if stats["sum"] is None else stats["sum"] + layer_sum
        stats["sum_squares"] += float(sum_squares.item())
        stats["tokens"] += token_count
        return output


class _SteeringLayer:
    def __init__(self, layer, vector, alpha):
        self.layer = layer
        self.vector = vector
        self.alpha = float(alpha)

    def __getattr__(self, name):
        return getattr(self.layer, name)

    def __call__(self, *args, **kwargs):
        mx = _mlx()
        output = self.layer(*args, **kwargs)
        hidden, kind = _get_hidden(output)
        if not hasattr(hidden, "shape") or len(hidden.shape) < 2:
            raise ValueError("decoder layer output must include a sequence axis")
        if hidden.shape[-2] == 1:
            if hidden.shape[-1] != self.vector.shape[0]:
                raise ValueError("steering vector width does not match decoder hidden size")
            delta = (self.alpha * self.vector).astype(hidden.dtype)
            hidden = hidden + delta
            return _replace_hidden(output, hidden, kind)
        return output


_STEERING_STATE = {}


def capture_means(model, tokenizer, rows, layers) -> dict:
    """Capture pooled means and residual RMS at answer positions, online.

    Each row has ``messages`` (system and user messages) and ``answer``. The
    assistant answer is appended as a final message using the same chat-template
    prefix check as ``route_nll.split_ids``.
    """
    mx = _mlx()
    decoder_layers = find_layers(model)
    layer_ids = _validate_layers(decoder_layers, layers)
    totals = {i: {"sum": None, "sum_squares": 0.0, "tokens": 0} for i in layer_ids}
    originals = {i: decoder_layers[i] for i in layer_ids}
    try:
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("messages"), list):
                raise ValueError("each capture row must contain a messages list")
            messages = row["messages"]
            answer = row.get("answer")
            if not isinstance(answer, str) or not answer:
                raise ValueError("each capture row must contain a nonempty answer string")
            if not messages or messages[-1].get("role") == "assistant":
                raise ValueError("capture messages must end before the assistant answer")
            prompt = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, enable_thinking=False
            )
            full_messages = messages + [{"role": "assistant", "content": answer}]
            full = tokenizer.apply_chat_template(
                full_messages, tokenize=True, enable_thinking=False
            )
            if list(full[:len(prompt)]) != list(prompt):
                raise ValueError("full chat tokens do not start with the generation prompt")
            if len(full) <= len(prompt):
                raise ValueError("assistant answer produced no tokens")
            start, stop = len(prompt), len(full)
            for layer_id in layer_ids:
                decoder_layers[layer_id] = _CaptureLayer(
                    originals[layer_id], layer_id, start, stop, totals
                )
            model(mx.array([full]))
            for layer_id in layer_ids:
                decoder_layers[layer_id] = originals[layer_id]
    finally:
        for layer_id, layer in originals.items():
            decoder_layers[layer_id] = layer

    if not rows:
        raise ValueError("capture rows must not be empty")
    means = {}
    residual_rms = {}
    for layer_id, stats in totals.items():
        if stats["tokens"] == 0 or stats["sum"] is None:
            raise ValueError(f"no answer activations captured for layer {layer_id}")
        means[layer_id] = (stats["sum"] / stats["tokens"]).astype(np.float32)
        width = means[layer_id].size
        residual_rms[layer_id] = math.sqrt(stats["sum_squares"] / (stats["tokens"] * width))
    return MeanCapture(means, residual_rms=residual_rms)


def build_directions(pref_means, unpref_means) -> tuple[np.ndarray, np.ndarray]:
    """Return preferred-minus-unpreferred directions and pooled residual RMS."""
    pref_keys = set(pref_means)
    unpref_keys = set(unpref_means)
    if not pref_keys or pref_keys != unpref_keys:
        raise ValueError("preferred and unpreferred means must have the same nonempty layer keys")
    if any(not isinstance(k, (int, np.integer)) for k in pref_keys):
        raise ValueError("mean keys must be integer layer indices")
    layer_ids = sorted(int(k) for k in pref_keys)
    directions = []
    rms_values = []
    pref_rms = getattr(pref_means, "residual_rms", {})
    unpref_rms = getattr(unpref_means, "residual_rms", {})
    for layer_id in layer_ids:
        preferred = np.asarray(pref_means[layer_id], dtype=np.float32)
        unpreferred = np.asarray(unpref_means[layer_id], dtype=np.float32)
        if preferred.ndim != 1 or preferred.shape != unpreferred.shape or preferred.size == 0:
            raise ValueError(f"invalid mean vector shape at layer {layer_id}")
        if not np.isfinite(preferred).all() or not np.isfinite(unpreferred).all():
            raise ValueError(f"means at layer {layer_id} must be finite")
        directions.append(preferred - unpreferred)
        if layer_id in pref_rms and layer_id in unpref_rms:
            rms = math.sqrt((float(pref_rms[layer_id]) ** 2 + float(unpref_rms[layer_id]) ** 2) / 2.0)
        else:
            rms = math.sqrt((float(np.mean(preferred.astype(np.float64) ** 2)) +
                             float(np.mean(unpreferred.astype(np.float64) ** 2))) / 2.0)
        if not math.isfinite(rms) or rms < 0:
            raise ValueError(f"invalid residual RMS at layer {layer_id}")
        rms_values.append(rms)
    return np.stack(directions).astype(np.float32), np.asarray(rms_values, dtype=np.float32)


def install_steering(model, layer, vector, alpha) -> None:
    """Install a reversible output hook that steers single-token decode calls."""
    layers = find_layers(model)
    if not isinstance(layer, (int, np.integer)) or isinstance(layer, bool):
        raise ValueError("layer must be an integer")
    layer = int(layer)
    if layer < 0 or layer >= len(layers):
        raise ValueError("layer index out of range")
    if id(model) in _STEERING_STATE:
        raise ValueError("steering is already installed on this model")
    if not isinstance(alpha, (int, float, np.number)) or not math.isfinite(float(alpha)):
        raise ValueError("alpha must be finite")
    vector = np.asarray(vector)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ValueError("steering vector must be a nonempty finite 1-D array")
    hidden_size = _hidden_size(model, layers[layer])
    if hidden_size is not None and vector.shape != (hidden_size,):
        raise ValueError(f"steering vector width {vector.size} != hidden size {hidden_size}")
    mx = _mlx()
    original = layers[layer]
    wrapper = _SteeringLayer(original, mx.array(vector.astype(np.float32)), float(alpha))
    layers[layer] = wrapper
    _STEERING_STATE[id(model)] = (layers, layer, original, wrapper)


def remove_steering(model) -> None:
    """Restore the original layer object after ``install_steering``."""
    state = _STEERING_STATE.pop(id(model), None)
    if state is None:
        return
    layers, layer, original, wrapper = state
    if layers[layer] is wrapper:
        layers[layer] = original


def random_directions(hidden: int, n: int, seed: int, norm: float) -> np.ndarray:
    """Create seeded Gaussian directions with exactly the requested L2 norm."""
    for name, value in (("hidden", hidden), ("n", n), ("seed", seed)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool):
            raise ValueError(f"{name} must be an integer")
    if hidden <= 0 or n <= 0 or seed < 0:
        raise ValueError("hidden and n must be positive and seed must be nonnegative")
    if not isinstance(norm, (int, float, np.number)) or not math.isfinite(float(norm)) or norm < 0:
        raise ValueError("norm must be finite and nonnegative")
    rng = np.random.default_rng(int(seed))
    values = rng.standard_normal((int(n), int(hidden))).astype(np.float64)
    lengths = np.linalg.norm(values, axis=1)
    values *= (float(norm) / lengths)[:, None]
    return values.astype(np.float32)


def _read_pairs(path, split):
    pairs = []
    with path.open() as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if row.get("split") == split:
                pairs.append(row)
    if not pairs:
        raise ValueError(f"no pairs for split {split!r} in {path}")
    return pairs


def _capture_cli(args):
    from mlx_lm import load

    mx = _mlx()
    # Wire the weights like mlx_lm.server does; otherwise macOS compresses them and capture stalls.
    mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    model, tokenizer = load(args.model)
    layers = find_layers(model)
    selected = sorted(_validate_layers(layers, args.layers))
    pairs = _read_pairs(args.pairs, args.split)
    preferred_rows, unpreferred_rows = [], []
    for pair in pairs:
        messages = pair.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"{pair.get('pair_id')}: expected [system, user] messages")
        preferred_rows.append({"messages": messages, "answer": pair["preferred"]})
        unpreferred_rows.append({"messages": messages, "answer": pair["unpreferred"]})
    preferred = capture_means(model, tokenizer, preferred_rows, selected)
    unpreferred = capture_means(model, tokenizer, unpreferred_rows, selected)
    directions, residual_rms = build_directions(preferred, unpreferred)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, layers=np.asarray(selected, dtype=np.int32),
                        directions=directions, residual_rms=residual_rms)
    print(f"captured {len(pairs)} pairs across {len(selected)} layers -> {args.out}")
    return 0


def _serve_cli(args, server_args):
    import mlx_lm.server

    archive = np.load(args.directions, allow_pickle=False)
    required = {"layers", "directions", "residual_rms"}
    if not required.issubset(archive.files):
        raise ValueError(f"direction archive must contain {sorted(required)}")
    layer_ids = np.asarray(archive["layers"], dtype=np.int64)
    directions = np.asarray(archive["directions"], dtype=np.float32)
    residual_rms = np.asarray(archive["residual_rms"], dtype=np.float32)
    if directions.ndim != 2 or layer_ids.shape != (directions.shape[0],) or residual_rms.shape != (directions.shape[0],):
        raise ValueError("direction archive arrays have inconsistent shapes")
    matches = np.flatnonzero(layer_ids == args.layer)
    if len(matches) != 1:
        raise ValueError(f"layer {args.layer} is not present exactly once in the archive")
    index = int(matches[0])
    direction = directions[index]
    direction_norm = float(np.linalg.norm(direction.astype(np.float64)))
    target_norm = float(residual_rms[index]) * math.sqrt(direction.size)
    if direction_norm == 0 or not math.isfinite(direction_norm) or target_norm <= 0:
        raise ValueError("selected direction and residual RMS must be nonzero")
    if args.random_seed is not None and args.random_seed < 0:
        raise ValueError("random seed must be nonnegative")
    vector = direction * (target_norm / direction_norm)
    if args.random_seed is not None:
        vector = random_directions(direction.size, 1, args.random_seed, target_norm)[0]

    provider_cls = mlx_lm.server.ModelProvider
    original_load = provider_cls.load
    patched_models = set()

    def patched_load(self, *call_args, **call_kwargs):
        result = original_load(self, *call_args, **call_kwargs)
        model = result[0]
        if id(model) not in patched_models:
            install_steering(model, args.layer, vector, args.alpha)
            patched_models.add(id(model))
        return result

    provider_cls.load = patched_load
    sys.argv = ["mlx_lm.server", *server_args]
    mlx_lm.server.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    _total_memory = mem_guard.total_memory_bytes()
    _headroom = mem_guard.headroom_bytes_from_env(os.environ)
    print(mem_guard.startup_limit_message(_total_memory, _headroom), flush=True)
    mem_guard.start(total_bytes=_total_memory, headroom_bytes=_headroom)
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capture = subparsers.add_parser("capture")
    capture.add_argument("--model", required=True)
    capture.add_argument("--pairs", type=Path, required=True)
    capture.add_argument("--split", choices=("build", "dev"), default="build")
    capture.add_argument("--layers", required=True, type=lambda value: [int(x) for x in value.split(",")])
    capture.add_argument("--out", type=Path, required=True)
    serve = subparsers.add_parser("serve")
    serve.add_argument("--directions", type=Path, required=True)
    serve.add_argument("--layer", type=int, required=True)
    serve.add_argument("--alpha", type=float, required=True)
    serve.add_argument("--random-seed", type=int)
    args, rest = parser.parse_known_args(argv)
    if args.command == "capture":
        return _capture_cli(args)
    return _serve_cli(args, rest)


if __name__ == "__main__":
    raise SystemExit(main())
