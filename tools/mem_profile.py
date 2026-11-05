#!/usr/bin/env python3
"""Measure the memory phases and layer waterfall of one MLX LoRA step.

The waterfall's forward-layer values are active MLX bytes immediately after
``mx.eval`` forces that layer's output in an instrumented value-and-grad pass. They describe the
live allocator state at that boundary, not that layer's isolated allocation or
the pass's peak; lazy evaluation and reuse can move work across boundaries.
Each backward-cut value is the peak during value-and-grad of the complete
masked loss after ``mx.stop_gradient`` is applied to outputs below the named
layer. It includes gradients for that layer and all higher layers, excludes
lower-layer gradients, and does not isolate one layer's marginal gradient cost.
Checkpoint recomputation, lazy execution, and allocator reuse limit layer-wise
attribution. Model loading mirrors mlx_lm/lora.py:347-352; LoRA setup mirrors
mlx_lm/lora.py:232-252; Adam setup mirrors :285-303. Checkpointing and training
mode mirror mlx_lm/tuner/trainer.py:27-39, :236-239, and :263; the update call
mirrors :258.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
import time
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if __package__:
    from . import mem_rows as row_tools
else:
    import mem_rows as row_tools

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
CONFIG_PATH = ROOT / "docs" / "finetune" / "lora.yaml"
DATA_PATH = row_tools.DATA_PATH
MODEL_PATH = "/Users/hatim/models/Qwen3.6-35B-A3B-4bit"
GIB = 2**30
build_row = row_tools.build_row
_row_source_name = row_tools.row_source_name


def _import_local_module(name: str) -> Any:
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    return importlib.import_module(name)


def _import_mlx() -> tuple[Any, Any, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    return mx, nn, optim


def load_training_config() -> dict[str, Any]:
    import yaml

    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("lora_parameters"), dict):
        raise ValueError(f"Invalid LoRA configuration in {CONFIG_PATH}")
    return config


def _row_arrays(row: Sequence[int], mx: Any) -> tuple[Any, Any]:
    batch = mx.array([row], dtype=mx.int32)
    lengths = mx.array([[len(row) // 2, len(row)]], dtype=mx.int32)
    return batch, lengths


def _training_input(
    tokens: int,
    tokenizer: Any,
    row_source: str | Path | Iterable[Mapping[str, Any]] | None,
    mx: Any,
) -> tuple[list[int], list[bool], Any, Any]:
    row, row_mask = build_row(tokens, tokenizer, row_source)
    batch, lengths = _row_arrays(row, mx)
    mx.eval(batch, lengths)
    return row, row_mask, batch, lengths


def _memory_measurement(mx: Any, phase: str, started: float) -> dict[str, Any]:
    return {
        "phase": phase,
        "peak_memory_bytes": int(mx.get_peak_memory()),
        "active_memory_bytes": int(mx.get_active_memory()),
        "cache_memory_bytes": int(mx.get_cache_memory()),
        "system_wired_bytes": _import_local_module("mem_guard").read_wired_bytes(),
        "wall_time_seconds": max(0.0, time.perf_counter() - started),
    }


def _cache_limit_bytes(limit_gb: float | None) -> int | None:
    if limit_gb is None:
        return None
    if not math.isfinite(limit_gb) or limit_gb < 0:
        raise ValueError("cache-limit-gb must be a finite float >= 0")
    limit_bytes = limit_gb * GIB
    if not math.isfinite(limit_bytes):
        raise ValueError("cache-limit-gb is too large")
    return int(limit_bytes)


def _cache_limit_config(limit_gb: float | None) -> float | str:
    _cache_limit_bytes(limit_gb)
    return "default" if limit_gb is None else float(limit_gb)


def _emit(
    config: Mapping[str, Any],
    measurement: Mapping[str, Any],
    callback: Callable[[dict[str, Any]], None] | None,
    **extra: Any,
) -> dict[str, Any]:
    record = {"config": dict(config), **dict(measurement), **extra}
    if callback is not None:
        callback(record)
    return record


def _load_model(
    gdn_checkpoint: int,
    model: Any | None = None,
    tokenizer: Any | None = None,
    require_tokenizer: bool = True,
    setup: Callable[[Any, Mapping[str, Any]], None] | None = None,
    training_config: dict[str, Any] | None = None,
    cache_limit_gb: float | None = None,
    on_runtime_ready: Callable[[], None] | None = None,
) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    cache_limit_bytes = _cache_limit_bytes(cache_limit_gb)
    if training_config is None:
        training_config = load_training_config()
    if model is None:
        import mem_guard

        mem_guard.start()
        import mlx.core as mx

        if cache_limit_bytes is not None:
            mx.set_cache_limit(cache_limit_bytes)
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
    else:
        import mlx.core as mx

        if cache_limit_bytes is not None:
            mx.set_cache_limit(cache_limit_bytes)
    if on_runtime_ready is not None:
        on_runtime_ready()
    mx.reset_peak_memory()
    started = time.perf_counter()
    if gdn_checkpoint:
        _import_local_module("gdn_checkpoint").install()
    if model is None:
        from mlx_lm.utils import load

        model, tokenizer = load(
            MODEL_PATH,
            tokenizer_config={"trust_remote_code": False},
            trust_remote_code=False,
        )
    if tokenizer is None and require_tokenizer:
        raise ValueError("tokenizer is required when injecting a model")
    if setup is not None:
        setup(model, training_config)
    mx.eval(model.parameters())
    load_measurement = _memory_measurement(mx, "load", started)
    return mx, model, tokenizer, load_measurement, training_config


def _prepare_lora(model: Any, layers: int, config: Mapping[str, Any]) -> None:
    if layers > len(model.layers):
        raise ValueError(f"Requested to train {layers} layers but model has {len(model.layers)}")
    mx, _, _ = _import_mlx()
    mx.random.seed(int(config.get("seed", 0)))
    model.freeze()
    from mlx_lm.tuner.utils import linear_to_lora_layers

    linear_to_lora_layers(model, layers, dict(config["lora_parameters"]))


def _prepare_training(
    model: Any,
    layers: int,
    config: Mapping[str, Any],
    prepared: bool,
    use_grad_checkpoint: bool,
) -> None:
    if layers > len(model.layers):
        raise ValueError(f"Requested to train {layers} layers but model has {len(model.layers)}")
    if not prepared:
        _prepare_lora(model, layers, config)
    if use_grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint

        grad_checkpoint(model.layers[0])
    model.train()


def _make_config(
    command: str,
    tokens: int,
    layers: int | None,
    gdn_checkpoint: int,
    grad_checkpoint: bool | None,
    injected: bool,
    **extra: Any,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "command": command,
        "tokens": tokens,
        "layers": layers,
        "gdn_checkpoint": gdn_checkpoint,
        "grad_checkpoint": grad_checkpoint,
        "model": "injected" if injected else MODEL_PATH,
        "run_id": uuid.uuid4().hex,
    }
    config.update(extra)
    return config


def _step_config(
    tokens: int,
    layers: int,
    gdn_checkpoint: int,
    grad_checkpoint: bool,
    injected: bool,
    capture: str | None,
    training_config: Mapping[str, Any],
    row_source: str | Path | Iterable[Mapping[str, Any]] | None,
    cache_limit_gb: float | None,
) -> dict[str, Any]:
    return _make_config(
        "step", tokens, layers, gdn_checkpoint, grad_checkpoint, injected,
        capture=capture, batch_size=1,
        learning_rate=float(training_config["learning_rate"]),
        lora_parameters=training_config["lora_parameters"],
        row_source=_row_source_name(row_source), completion_mask="last_half",
        cache_limit_gb=_cache_limit_config(cache_limit_gb),
    )


def _measure_forward(
    mx: Any, loss_fn: Callable[..., Any], model: Any, batch: Any, lengths: Any
) -> tuple[dict[str, Any], Any, Any]:
    mx.reset_peak_memory()
    started = time.perf_counter()
    loss, ntokens = loss_fn(model, batch, lengths)
    mx.eval(loss, ntokens)
    return _memory_measurement(mx, "forward", started), loss, ntokens


def _measure_value_and_grad(
    mx: Any,
    value_and_grad: Callable[..., Any],
    model: Any,
    batch: Any,
    lengths: Any,
    capture: str | None,
) -> tuple[dict[str, Any], Any, Any, Any]:
    mx.reset_peak_memory()
    started = time.perf_counter()
    capturing = False
    try:
        if capture is not None:
            mx.metal.start_capture(capture)
            capturing = True
        (loss, ntokens), gradients = value_and_grad(model, batch, lengths)
        mx.eval(loss, ntokens, gradients)
    finally:
        if capturing:
            mx.metal.stop_capture()
    return _memory_measurement(mx, "value_and_grad", started), loss, ntokens, gradients


def _measure_optimizer_update(
    mx: Any, optimizer: Any, model: Any, gradients: Any
) -> dict[str, Any]:
    mx.reset_peak_memory()
    started = time.perf_counter()
    optimizer.update(model, gradients)
    mx.eval(model.parameters(), optimizer.state)
    return _memory_measurement(mx, "optimizer_update", started)


def _load_step_context(
    tokens: int,
    layers: int,
    gdn_checkpoint: int,
    grad_checkpoint: bool | None,
    capture: str | None,
    cache_limit_gb: float | None,
    model: Any | None,
    tokenizer: Any | None,
    row_source: str | Path | Iterable[Mapping[str, Any]] | None,
    model_prepared: bool,
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any], dict[str, Any]]:
    injected = model is not None
    training_config = load_training_config()
    use_checkpoint = training_config["grad_checkpoint"] if grad_checkpoint is None else grad_checkpoint
    prepared = model is not None and model_prepared
    setup = lambda selected, config: _prepare_training(
        selected, layers, config, prepared, use_checkpoint
    )
    mx, model, tokenizer, load_metrics, training_config = _load_model(
        gdn_checkpoint, model, tokenizer, setup=setup, training_config=training_config,
        cache_limit_gb=cache_limit_gb,
    )
    run_config = _step_config(
        tokens, layers, gdn_checkpoint, bool(use_checkpoint), injected,
        capture, training_config, row_source, cache_limit_gb,
    )
    return mx, model, tokenizer, training_config, load_metrics, run_config


def run_step(
    tokens: int,
    layers: int,
    *,
    gdn_checkpoint: int = 1,
    grad_checkpoint: bool | None = None,
    capture: str | None = None,
    cache_limit_gb: float | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    row_source: str | Path | Iterable[Mapping[str, Any]] | None = None,
    model_prepared: bool = True,
    on_measurement: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if tokens < 2 or layers < 1 or gdn_checkpoint not in (0, 1):
        raise ValueError("tokens must be >= 2, layers >= 1, and gdn_checkpoint must be 0 or 1")
    if capture is not None and os.environ.get("MTL_CAPTURE_ENABLED") != "1":
        raise RuntimeError("--capture requires MTL_CAPTURE_ENABLED=1")
    mx, model, tokenizer, training_config, load_metrics, run_config = _load_step_context(
        tokens, layers, gdn_checkpoint, grad_checkpoint, capture, cache_limit_gb,
        model, tokenizer, row_source, model_prepared,
    )
    phases: dict[str, dict[str, Any]] = {}
    phases["load"] = _emit(run_config, load_metrics, on_measurement)
    row, row_mask, batch, lengths = _training_input(tokens, tokenizer, row_source, mx)
    loss_fn = _import_local_module("ft_lowmem_loss").completion_only_loss
    mx, _, optim = _import_mlx()
    optimizer = optim.Adam(learning_rate=float(training_config["learning_rate"]))
    forward_metrics, forward_loss, forward_ntokens = _measure_forward(
        mx, loss_fn, model, batch, lengths
    )
    phases["forward"] = _emit(
        run_config, forward_metrics, on_measurement,
        loss=float(forward_loss.item()), ntokens=int(forward_ntokens.item()),
    )
    return _finish_step(
        mx, model, optimizer, loss_fn, batch, lengths, forward_loss,
        forward_ntokens, phases, run_config, on_measurement, capture, row, row_mask,
    )


def _finish_step(
    mx: Any,
    model: Any,
    optimizer: Any,
    loss_fn: Callable[..., Any],
    batch: Any,
    lengths: Any,
    forward_loss: Any,
    forward_ntokens: Any,
    phases: dict[str, dict[str, Any]],
    config: Mapping[str, Any],
    callback: Callable[[dict[str, Any]], None] | None,
    capture: str | None,
    row: list[int],
    row_mask: list[bool],
) -> dict[str, Any]:
    import mlx.nn as nn

    value_and_grad = nn.value_and_grad(model, loss_fn)
    grad_metrics, loss, ntokens, gradients = _measure_value_and_grad(
        mx, value_and_grad, model, batch, lengths, capture
    )
    phases["value_and_grad"] = _emit(
        config, grad_metrics, callback,
        loss=float(loss.item()), ntokens=int(ntokens.item()),
    )
    update_metrics = _measure_optimizer_update(mx, optimizer, model, gradients)
    phases["optimizer_update"] = _emit(config, update_metrics, callback)
    return {
        "config": dict(config),
        "phases": phases,
        "loss": float(loss.item()),
        "forward_loss": float(forward_loss.item()),
        "ntokens": int(ntokens.item()),
        "batch": batch,
        "lengths": lengths,
        "row": row,
        "row_mask": row_mask,
    }


@contextmanager
def _instrument_layers(
    model: Any,
    mx: Any,
    observer: Callable[[int], None] | None = None,
    cut_below: int | None = None,
) -> Iterator[None]:
    layers = model.layers
    layer_indices = {id(layer): index for index, layer in enumerate(layers)}
    layer_type = type(layers[0])
    original = layer_type.__call__

    def wrapped(layer: Any, *args: Any, **kwargs: Any) -> Any:
        output = original(layer, *args, **kwargs)
        index = layer_indices.get(id(layer))
        if index is not None and cut_below is not None and index < cut_below:
            output = mx.stop_gradient(output)
        if index is not None and observer is not None:
            mx.eval(output)
            observer(index)
        return output

    layer_type.__call__ = wrapped
    try:
        yield
    finally:
        layer_type.__call__ = original


def _measure_cut(
    mx: Any,
    value_and_grad: Callable[..., Any],
    model: Any,
    batch: Any,
    lengths: Any,
    cut_below: int,
) -> tuple[dict[str, Any], float, int]:
    mx.reset_peak_memory()
    started = time.perf_counter()
    with _instrument_layers(model, mx, cut_below=cut_below):
        (loss, ntokens), gradients = value_and_grad(model, batch, lengths)
        mx.eval(loss, ntokens, gradients)
    metrics = _memory_measurement(mx, "backward_cut", started)
    return metrics, float(loss.item()), int(ntokens.item())


def run_waterfall(
    tokens: int,
    layers: int,
    *,
    gdn_checkpoint: int = 1,
    cache_limit_gb: float | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    row_source: str | Path | Iterable[Mapping[str, Any]] | None = None,
    model_prepared: bool = True,
    on_measurement: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if tokens < 2 or layers < 1 or gdn_checkpoint not in (0, 1):
        raise ValueError("tokens must be >= 2, layers >= 1, and gdn_checkpoint must be 0 or 1")
    injected = model is not None
    training_config = load_training_config()
    use_checkpoint = bool(training_config["grad_checkpoint"])
    prepared = model is not None and model_prepared
    setup = lambda selected, config: _prepare_training(
        selected, layers, config, prepared, use_checkpoint
    )
    mx, model, tokenizer, load_metrics, training_config = _load_model(
        gdn_checkpoint, model, tokenizer, setup=setup, training_config=training_config,
        cache_limit_gb=cache_limit_gb,
    )
    run_config = _make_config(
        "waterfall", tokens, layers, gdn_checkpoint, use_checkpoint, injected,
        batch_size=1, learning_rate=float(training_config["learning_rate"]),
        lora_parameters=training_config["lora_parameters"],
        row_source=_row_source_name(row_source),
        completion_mask="last_half",
        cache_limit_gb=_cache_limit_config(cache_limit_gb),
    )
    measurements = [_emit(run_config, load_metrics, on_measurement)]
    row, _ = build_row(tokens, tokenizer, row_source)
    batch, lengths = _row_arrays(row, mx)
    mx.eval(batch, lengths)
    loss_fn = _import_local_module("ft_lowmem_loss").completion_only_loss
    measurements.extend(_waterfall_forward(mx, model, loss_fn, batch, lengths, run_config, on_measurement))
    measurements.extend(_waterfall_cuts(mx, model, loss_fn, batch, lengths, layers, run_config, on_measurement))
    return {"config": run_config, "measurements": measurements}


def _waterfall_forward(
    mx: Any,
    model: Any,
    loss_fn: Callable[..., Any],
    batch: Any,
    lengths: Any,
    config: Mapping[str, Any],
    callback: Callable[[dict[str, Any]], None] | None,
) -> list[dict[str, Any]]:
    import mlx.nn as nn

    layer_values: list[dict[str, Any]] = []
    started = time.perf_counter()

    def observe(index: int) -> None:
        layer_values.append({
            "layer_index": index,
            "peak_memory_bytes": int(mx.get_peak_memory()),
            "active_memory_bytes": int(mx.get_active_memory()),
            "cache_memory_bytes": int(mx.get_cache_memory()),
            "system_wired_bytes": _import_local_module("mem_guard").read_wired_bytes(),
            "wall_time_seconds": max(0.0, time.perf_counter() - started),
        })

    mx.reset_peak_memory()
    value_and_grad = nn.value_and_grad(model, loss_fn)
    with _instrument_layers(model, mx, observer=observe):
        (loss, ntokens), gradients = value_and_grad(model, batch, lengths)
        mx.eval(loss, ntokens, gradients)
    records = [
        _emit(config, {"phase": "forward_layer", **values}, callback)
        for values in layer_values
    ]
    records.append(_emit(config, _memory_measurement(mx, "forward", started), callback))
    return records


def _waterfall_cuts(
    mx: Any,
    model: Any,
    loss_fn: Callable[..., Any],
    batch: Any,
    lengths: Any,
    layers: int,
    config: Mapping[str, Any],
    callback: Callable[[dict[str, Any]], None] | None,
) -> list[dict[str, Any]]:
    import mlx.nn as nn

    first_lora_layer = len(model.layers) - layers
    value_and_grad = nn.value_and_grad(model, loss_fn)
    records = []
    for cut_below in range(len(model.layers) - 1, first_lora_layer - 1, -1):
        metrics, loss, ntokens = _measure_cut(
            mx, value_and_grad, model, batch, lengths, cut_below
        )
        records.append(_emit(
            config, metrics, callback,
            cut_below_layer=cut_below, loss=loss, ntokens=ntokens,
        ))
    return records


def _scope_layer(nn: Any, layer: Any) -> Any:
    class LayerScope(nn.Module):
        def __init__(self, selected: Any) -> None:
            super().__init__()
            self.layers = [selected]

    return LayerScope(layer)


def _block_component(layer: Any, kind: str) -> Any:
    attribute = {"moe": "mlp", "gdn": "linear_attn", "attention": "self_attn"}[kind]
    if not hasattr(layer, attribute):
        raise ValueError(f"layer does not contain a {kind} block")
    return getattr(layer, attribute)


def _call_block(block: Any, kind: str, inputs: Any) -> Any:
    if kind == "moe":
        return block(inputs)
    return block(inputs, mask=None, cache=None)


def _measure_block_grad(
    mx: Any, block: Any, kind: str, inputs: Any, use_lora: bool
) -> dict[str, Any]:
    mx.eval(inputs)
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    started = time.perf_counter()
    if use_lora:
        def objective(parameters: Any, values: Any) -> Any:
            block.update(parameters)
            return _call_block(block, kind, values).sum()

        value, (parameter_gradients, input_gradients) = mx.value_and_grad(
            objective, argnums=(0, 1)
        )(block.trainable_parameters(), inputs)
        mx.eval(value, parameter_gradients, input_gradients)
    else:
        value, input_gradients = mx.value_and_grad(
            lambda values: _call_block(block, kind, values).sum()
        )(inputs)
        mx.eval(value, input_gradients)
    metrics = _memory_measurement(mx, "value_and_grad", started)
    metrics["baseline_active_memory_bytes"] = baseline
    metrics["incremental_peak_memory_bytes"] = max(0, metrics["peak_memory_bytes"] - baseline)
    return metrics


def run_block(
    kind: str,
    layer_index: int,
    tokens: int,
    *,
    use_lora: bool = False,
    cache_limit_gb: float | None = None,
    model: Any | None = None,
    tokenizer: Any | None = None,
    on_measurement: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if kind not in ("moe", "gdn", "attention") or tokens < 1 or layer_index < 0:
        raise ValueError("kind, tokens, or layer_index is invalid")
    injected = model is not None
    mx, nn, model, load_metrics, training_config = _load_block_model(
        model, tokenizer, cache_limit_gb
    )
    if layer_index >= len(model.layers):
        raise ValueError(f"layer-index must be below {len(model.layers)}")
    model.freeze()
    scope = _scope_layer(nn, model.layers[layer_index])
    if use_lora:
        from mlx_lm.tuner.utils import linear_to_lora_layers

        mx.random.seed(int(training_config.get("seed", 0)))
        linear_to_lora_layers(scope, 1, dict(training_config["lora_parameters"]))
    layer = scope.layers[0]
    block = _block_component(layer, kind)
    model.train()
    dtype = layer.input_layernorm.weight.dtype
    inputs = mx.random.normal((1, tokens, 2048), dtype=dtype)
    run_config = _make_config(
        "block", tokens, None, 0, False, injected,
        kind=kind, layer_index=layer_index, lora=int(use_lora),
        lora_parameters=training_config["lora_parameters"] if use_lora else None,
        cache_limit_gb=_cache_limit_config(cache_limit_gb),
    )
    load_record = _emit(run_config, load_metrics, on_measurement)
    forward_metrics, grad_metrics = _measure_block(mx, block, kind, inputs, use_lora)
    forward_record = _emit(run_config, forward_metrics, on_measurement)
    grad_record = _emit(run_config, grad_metrics, on_measurement)
    return {
        "config": run_config,
        "load": load_record,
        "forward": forward_record,
        "value_and_grad": grad_record,
    }


def _load_block_model(
    model: Any | None, tokenizer: Any | None, cache_limit_gb: float | None
) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    mx, model, _, load_metrics, training_config = _load_model(
        0, model, tokenizer, require_tokenizer=False, cache_limit_gb=cache_limit_gb
    )
    _, nn, _ = _import_mlx()
    return mx, nn, model, load_metrics, training_config


def _measure_block(
    mx: Any, block: Any, kind: str, inputs: Any, use_lora: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    mx.eval(inputs)
    baseline = int(mx.get_active_memory())
    mx.reset_peak_memory()
    started = time.perf_counter()
    output = _call_block(block, kind, inputs)
    mx.eval(output)
    forward = _memory_measurement(mx, "forward", started)
    forward["baseline_active_memory_bytes"] = baseline
    forward["incremental_peak_memory_bytes"] = max(0, forward["peak_memory_bytes"] - baseline)
    del output
    gradients = _measure_block_grad(mx, block, kind, inputs, use_lora)
    return forward, gradients


def _measurement_writer(path: str | Path) -> Callable[[dict[str, Any]], None]:
    append = _import_local_module("mem_sweep").append_jsonl
    return lambda record: append(path, record)


def run_trainer_step(
    tokens: int,
    layers: int,
    **kwargs: Any,
) -> dict[str, Any]:
    return _import_local_module("mem_trainer").run_trainer_step(
        tokens, layers, **kwargs
    )


def _parse_cache_limit_gb(value: str) -> float:
    try:
        parsed = float(value)
        _cache_limit_bytes(parsed)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile MLX LoRA training memory")
    commands = parser.add_subparsers(dest="command", required=True)
    sweep_tools = _import_local_module("mem_sweep")
    step = commands.add_parser("step")
    step.add_argument("--tokens", type=int, required=True)
    step.add_argument("--layers", type=int, required=True)
    step.add_argument("--gdn-checkpoint", type=int, choices=(0, 1), default=1)
    step.add_argument("--no-grad-checkpoint", action="store_true")
    step.add_argument("--capture")
    step.add_argument("--cache-limit-gb", type=_parse_cache_limit_gb)
    step.add_argument("--out", required=True)
    waterfall = commands.add_parser("waterfall")
    waterfall.add_argument("--tokens", type=int, required=True)
    waterfall.add_argument("--layers", type=int, required=True)
    waterfall.add_argument("--cache-limit-gb", type=_parse_cache_limit_gb)
    waterfall.add_argument("--out", required=True)
    block = commands.add_parser("block")
    block.add_argument("--kind", choices=("moe", "gdn", "attention"), required=True)
    block.add_argument("--layer-index", type=int, required=True)
    block.add_argument("--tokens", type=int, required=True)
    block.add_argument("--lora", type=int, choices=(0, 1), default=0)
    block.add_argument("--cache-limit-gb", type=_parse_cache_limit_gb)
    block.add_argument("--out", required=True)
    trainer_step = commands.add_parser("trainer-step")
    trainer_step.add_argument("--tokens", type=int, required=True)
    trainer_step.add_argument("--layers", type=int, required=True)
    trainer_step.add_argument("--cache-limit-gb", type=_parse_cache_limit_gb)
    trainer_step.add_argument("--out", required=True)
    sweep = commands.add_parser("sweep")
    sweep.add_argument("--tokens", type=sweep_tools.parse_csv_ints, default=sweep_tools.parse_csv_ints("256,512,1000,2000,3000"))
    sweep.add_argument("--layers", type=sweep_tools.parse_csv_ints, default=sweep_tools.parse_csv_ints("1,2,3,4"))
    sweep.add_argument("--repeat", type=int, default=1)
    sweep.add_argument("--mode", choices=("step", "trainer-step"), default="trainer-step")
    sweep.add_argument("--cache-limit-gb", type=_parse_cache_limit_gb)
    sweep.add_argument("--out", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--in", dest="input_path", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    writer = _measurement_writer(args.out) if args.command != "fit" else None
    if args.command == "step":
        config = load_training_config()
        grad_checkpoint = bool(config["grad_checkpoint"] and not args.no_grad_checkpoint)
        run_step(
            args.tokens, args.layers, gdn_checkpoint=args.gdn_checkpoint,
            grad_checkpoint=grad_checkpoint, capture=args.capture,
            cache_limit_gb=args.cache_limit_gb,
            on_measurement=writer,
        )
    elif args.command == "waterfall":
        run_waterfall(
            args.tokens, args.layers, cache_limit_gb=args.cache_limit_gb,
            on_measurement=writer,
        )
    elif args.command == "block":
        run_block(
            args.kind, args.layer_index, args.tokens, use_lora=bool(args.lora),
            cache_limit_gb=args.cache_limit_gb,
            on_measurement=writer,
        )
    elif args.command == "trainer-step":
        run_trainer_step(
            args.tokens, args.layers, cache_limit_gb=args.cache_limit_gb,
            on_measurement=writer,
        )
    elif args.command == "sweep":
        _import_local_module("mem_sweep").run_sweep(
            args.tokens, args.layers, args.repeat, args.out,
            mode=args.mode, cache_limit_gb=args.cache_limit_gb,
        )
    else:
        sweep_tools = _import_local_module("mem_sweep")
        sweep_tools.print_fit(sweep_tools.fit_coefficients(
            sweep_tools.read_jsonl(args.input_path)
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
