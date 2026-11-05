#!/usr/bin/env python3
"""Source-derived memory estimate for one batch-one Qwen3.5 MoE LoRA step."""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import resource
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mem_budget_terms import (
    BOOL_BYTES,
    DEFAULT_DEVICE_MEMORY_GIB,
    DEFAULT_GDN_CHUNK,
    DEFAULT_LOSS_CHUNK,
    DEFAULT_MAX_RECOMMENDED_WORKING_SET_GIB,
    DEFAULT_OS_WIRED_GIB,
    DEFAULT_RANK,
    F32_BYTES,
    GIB,
    GDN_VJP_STATE_ARRAYS,
    I32_BYTES,
    MEM_GUARD_LIMIT_GIB,
    PREFLIGHT_MARGIN_GIB,
    attention_vjp_terms,
    block_incremental_bytes,
    budget_terms,
    gdn_state_bytes,
    layer_phase_terms,
    lora_dimensions,
    quantized_payload,
)


@dataclass(frozen=True)
class Shapes:
    batch_size: int
    hidden: int
    vocab: int
    layers: int
    layer_types: tuple[str, ...]
    attention_heads: int
    kv_heads: int
    head_dim: int
    experts: int
    top_k: int
    moe_intermediate: int
    shared_intermediate: int
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_key_dim: int
    gdn_value_dim: int
    gdn_conv_kernel: int
    quant_bits: int
    quant_group: int
    router_bits: int
    dtype_bytes: int
    lora_bytes: int
    optimizer_bytes: int
    gdn_checkpoint_chunk: int
    loss_chunk: int
    attention_vjp_block: int
    index_bytes: int
    bool_bytes: int
    weights_bytes: int
    weights_payload_bytes: int
    weights_dropped_bytes: int
    weights_exact: bool
    tie_word_embeddings: bool

    @property
    def gdn_key_width(self) -> int:
        return self.gdn_key_heads * self.gdn_key_dim

    @property
    def gdn_value_width(self) -> int:
        return self.gdn_value_heads * self.gdn_value_dim

    @property
    def gdn_conv_width(self) -> int:
        return 2 * self.gdn_key_width + self.gdn_value_width


@dataclass(frozen=True)
class AllocatorLimits:
    cache_cap_bytes: int
    gc_threshold_bytes: int
    occupancy_bytes: int
    recommended_working_set_bytes: int
    device_memory_bytes: int
    from_runtime: bool
    runtime_note: str


def _required(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise KeyError(f"missing config key: {path}.{key}")
    return mapping[key]


def _is_sanitized_away(
    name: str, model_type: str, tie_word_embeddings: bool
) -> bool:
    if "mtp." in name or (tie_word_embeddings and name == "lm_head.weight"):
        return True
    return model_type == "qwen3_5_moe" and (
        name.startswith("vision_tower") or name.startswith("model.visual")
    )


def _safe_tensor_payload(
    directory: Path, *, model_type: str, tie_word_embeddings: bool
) -> tuple[int, int, int] | None:
    files = sorted(directory.glob("model*.safetensors"))
    if not files:
        return None
    payload = resident = dropped = 0
    for path in files:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            header_size = struct.unpack("<Q", prefix)[0]
            header = json.loads(stream.read(header_size))
        for name, tensor in header.items():
            if name == "__metadata__":
                continue
            start, end = tensor["data_offsets"]
            size = end - start
            payload += size
            if _is_sanitized_away(name, model_type, tie_word_embeddings):
                dropped += size
            else:
                resident += size
    return payload, resident, dropped


def _estimated_weight_bytes(shapes: Shapes) -> int:
    h = shapes.hidden
    q = 2 * shapes.attention_heads * shapes.head_dim
    kv = shapes.kv_heads * shapes.head_dim
    key, value = shapes.gdn_key_width, shapes.gdn_value_width
    q4 = lambda size: quantized_payload(
        size, shapes.quant_bits, shapes.quant_group, shapes.dtype_bytes
    )
    qrouter = lambda size: quantized_payload(
        size, shapes.router_bits, shapes.quant_group, shapes.dtype_bytes
    )
    total = q4(2 * shapes.vocab * h)
    for kind in shapes.layer_types:
        if kind == "full_attention":
            total += q4(h * (q + 2 * kv) + q * h)
            total += 2 * shapes.head_dim * shapes.dtype_bytes
        else:
            total += q4(h * (2 * key + 2 * value + 2 * shapes.gdn_value_heads)
                        + value * h)
            total += shapes.gdn_conv_width * shapes.gdn_conv_kernel * shapes.dtype_bytes
            total += shapes.gdn_value_heads * (2 * F32_BYTES + shapes.dtype_bytes)
        total += q4(
            shapes.experts * 3 * h * shapes.moe_intermediate
            + 3 * h * shapes.shared_intermediate
        )
        total += qrouter(h * shapes.experts + h)
        total += 3 * h * shapes.dtype_bytes
    return total + (h + 2 * shapes.gdn_value_heads) * shapes.dtype_bytes


def _router_bits(quant: dict[str, Any]) -> int:
    for key, item in quant.items():
        if key.endswith(".mlp.gate") and isinstance(item, dict):
            return int(_required(item, "bits", "router override"))
    raise KeyError("missing config key: quantization_config.<layer>.mlp.gate")


def _config_values(root: dict[str, Any]) -> dict[str, Any]:
    text = _required(root, "text_config", "root")
    quant = _required(root, "quantization_config", "root")
    layers = int(_required(text, "num_hidden_layers", "text_config"))
    layer_types = tuple(_required(text, "layer_types", "text_config"))
    if len(layer_types) != layers:
        raise ValueError("text_config.layer_types length does not match num_hidden_layers")
    if any(kind not in {"linear_attention", "full_attention"} for kind in layer_types):
        raise ValueError("text_config.layer_types contains an unknown layer type")
    dtype = _required(text, "dtype", "text_config")
    dtype_bytes = {"float16": 2, "bfloat16": 2, "float32": 4}.get(dtype)
    if dtype_bytes is None:
        raise ValueError(f"unsupported text_config.dtype: {dtype}")
    return {
        "batch_size": 1,
        "hidden": int(_required(text, "hidden_size", "text_config")),
        "vocab": int(_required(text, "vocab_size", "text_config")),
        "layers": layers,
        "layer_types": layer_types,
        "attention_heads": int(_required(text, "num_attention_heads", "text_config")),
        "kv_heads": int(_required(text, "num_key_value_heads", "text_config")),
        "head_dim": int(_required(text, "head_dim", "text_config")),
        "experts": int(_required(text, "num_experts", "text_config")),
        "top_k": int(_required(text, "num_experts_per_tok", "text_config")),
        "moe_intermediate": int(_required(text, "moe_intermediate_size", "text_config")),
        "shared_intermediate": int(_required(text, "shared_expert_intermediate_size", "text_config")),
        "gdn_key_heads": int(_required(text, "linear_num_key_heads", "text_config")),
        "gdn_value_heads": int(_required(text, "linear_num_value_heads", "text_config")),
        "gdn_key_dim": int(_required(text, "linear_key_head_dim", "text_config")),
        "gdn_value_dim": int(_required(text, "linear_value_head_dim", "text_config")),
        "gdn_conv_kernel": int(_required(text, "linear_conv_kernel_dim", "text_config")),
        "quant_bits": int(_required(quant, "bits", "quantization_config")),
        "quant_group": int(_required(quant, "group_size", "quantization_config")),
        "router_bits": _router_bits(quant),
        "dtype_bytes": dtype_bytes,
        "lora_bytes": F32_BYTES,
        "optimizer_bytes": F32_BYTES,
        "gdn_checkpoint_chunk": DEFAULT_GDN_CHUNK,
        "loss_chunk": DEFAULT_LOSS_CHUNK,
        "attention_vjp_block": 1024,
        "index_bytes": I32_BYTES,
        "bool_bytes": BOOL_BYTES,
        "tie_word_embeddings": bool(_required(text, "tie_word_embeddings", "text_config")),
    }


def load_shapes(config_path: str | Path) -> Shapes:
    """Load every dimension used by the formulas; absent config keys are errors."""
    path = Path(config_path)
    root = json.loads(path.read_text(encoding="utf-8"))
    model_type = str(_required(root, "model_type", "root"))
    raw = _config_values(root)
    draft = Shapes(
        **raw, weights_bytes=0, weights_payload_bytes=0,
        weights_dropped_bytes=0, weights_exact=False,
    )
    payload = _safe_tensor_payload(
        path.parent, model_type=model_type,
        tie_word_embeddings=draft.tie_word_embeddings,
    )
    if payload is None:
        estimate = _estimated_weight_bytes(draft)
        return Shapes(
            **raw, weights_bytes=estimate, weights_payload_bytes=estimate,
            weights_dropped_bytes=0, weights_exact=False,
        )
    total, resident, dropped = payload
    return Shapes(
        **raw, weights_bytes=resident, weights_payload_bytes=total,
        weights_dropped_bytes=dropped, weights_exact=True,
    )


def _fallback_allocator_limits() -> AllocatorLimits:
    recommended = int(DEFAULT_MAX_RECOMMENDED_WORKING_SET_GIB * GIB)
    memory = int(DEFAULT_DEVICE_MEMORY_GIB * GIB)
    cap = int(min(1.5 * recommended, 0.95 * memory))
    gc = int(min(0.95 * recommended, cap))
    return AllocatorLimits(
        cap, gc, 0, recommended, memory, False,
        "MLX runtime device info unavailable; using source-era device constants.",
    )


def _disable_core_dumps() -> None:
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


@functools.lru_cache(maxsize=1)
def _runtime_device_values() -> tuple[int, int, int, str] | None:
    code = (
        "import json, mlx.core as mx; d=mx.device_info(); "
        "print(json.dumps([d['max_recommended_working_set_size'], "
        "d['memory_size'], mx.get_cache_memory()]))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code], check=False, capture_output=True,
            text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            preexec_fn=_disable_core_dumps,
        )
    except OSError as exc:
        return (0, 0, 0, f"MLX runtime query failed: {exc}")
    if result.returncode != 0:
        return (0, 0, 0, f"MLX runtime query exited {result.returncode}; using fallback values.")
    values = json.loads(result.stdout)
    if len(values) != 3 or any(int(value) < 0 for value in values):
        raise ValueError("mlx.core returned invalid device or cache memory values")
    return tuple(map(int, values)) + ("",)


def allocator_limits(cache_limit_gib: float | None = None) -> AllocatorLimits:
    """Read MLX device/cache runtime values, using documented fallbacks if absent."""
    runtime = _runtime_device_values()
    if runtime is None:
        limits = _fallback_allocator_limits()
    else:
        recommended, memory, _, note = runtime
        if note:
            fallback = _fallback_allocator_limits()
            limits = AllocatorLimits(
                fallback.cache_cap_bytes, fallback.gc_threshold_bytes, 0,
                fallback.recommended_working_set_bytes, fallback.device_memory_bytes,
                False, note,
            )
        else:
            try:
                import mlx.core as mx
            except ImportError:
                fallback = _fallback_allocator_limits()
                limits = AllocatorLimits(
                    fallback.cache_cap_bytes, fallback.gc_threshold_bytes, 0,
                    fallback.recommended_working_set_bytes,
                    fallback.device_memory_bytes, False,
                    "MLX runtime was unavailable in this process; using fallback values.",
                )
            else:
                device = mx.device_info()
                recommended = int(device["max_recommended_working_set_size"])
                memory = int(device["memory_size"])
                occupancy = int(mx.get_cache_memory())
                cap = int(min(1.5 * recommended, 0.95 * memory))
                gc = int(min(0.95 * recommended, cap))
                limits = AllocatorLimits(
                    cap, gc, occupancy, recommended, memory, True, ""
                )
    if cache_limit_gib is None:
        return limits
    if not math.isfinite(cache_limit_gib) or cache_limit_gib < 0:
        raise ValueError("cache_limit_gib must be finite and nonnegative")
    return AllocatorLimits(
        int(cache_limit_gib * GIB), limits.gc_threshold_bytes,
        limits.occupancy_bytes, limits.recommended_working_set_bytes,
        limits.device_memory_bytes, limits.from_runtime, limits.runtime_note,
    )


def default_cache_limit_bytes() -> int:
    return allocator_limits().cache_cap_bytes


def default_gc_limit_bytes() -> int:
    return allocator_limits().gc_threshold_bytes


def system_wired_bytes(
    active_peak_bytes: int,
    cache_limit_gib: float | None = None,
    os_wired_gib: float = DEFAULT_OS_WIRED_GIB,
) -> int:
    """Return a capacity-bound wired estimate: active + cache cap + OS reserve."""
    if active_peak_bytes < 0 or not math.isfinite(os_wired_gib) or os_wired_gib < 0:
        raise ValueError("memory values must be finite and nonnegative")
    cap = allocator_limits(cache_limit_gib).cache_cap_bytes
    return active_peak_bytes + cap + int(os_wired_gib * GIB)


def system_wired_occupancy_bytes(
    active_peak_bytes: int,
    cache_occupancy_bytes: int,
    os_wired_gib: float = DEFAULT_OS_WIRED_GIB,
) -> int:
    if min(active_peak_bytes, cache_occupancy_bytes) < 0:
        raise ValueError("memory values must be nonnegative")
    if not math.isfinite(os_wired_gib) or os_wired_gib < 0:
        raise ValueError("os_wired_gib must be finite and nonnegative")
    return active_peak_bytes + cache_occupancy_bytes + int(os_wired_gib * GIB)


def system_wired_floor_bytes(
    active_peak_bytes: int, os_wired_gib: float = DEFAULT_OS_WIRED_GIB
) -> int:
    return system_wired_occupancy_bytes(active_peak_bytes, 0, os_wired_gib)


def system_wired_prediction(
    active_peak_bytes: int, cache_limit_gib: float | None = None
) -> dict[str, int]:
    limits = allocator_limits(cache_limit_gib)
    return {
        "system_wired_current_occupancy": system_wired_occupancy_bytes(
            active_peak_bytes, limits.occupancy_bytes
        ),
        "system_wired_cache_cap_bound": system_wired_bytes(
            active_peak_bytes, cache_limit_gib
        ),
    }


def budget(
    shapes: Shapes,
    tokens: int,
    lora_layers: int,
    *,
    grad_checkpoint: bool = True,
    rank: int = DEFAULT_RANK,
    cache_limit_gib: float | None = None,
    assume_working_gdn_checkpoint: bool = False,
) -> dict[str, int]:
    """Return per-term bytes, including allocator diagnostics but not summable bounds."""
    terms = budget_terms(
        shapes, tokens, lora_layers, grad_checkpoint=grad_checkpoint,
        rank=rank, working_gdn_checkpoint=assume_working_gdn_checkpoint,
    )
    limits = allocator_limits(cache_limit_gib)
    terms.update({
        "weights_resident": shapes.weights_bytes,
        "weights_safetensors_payload": shapes.weights_payload_bytes,
        "weights_dropped_by_sanitize": shapes.weights_dropped_bytes,
        "allocator_cache_cap": limits.cache_cap_bytes,
        "allocator_gc_threshold": limits.gc_threshold_bytes,
        "allocator_cache_occupancy_now": limits.occupancy_bytes,
        "trainer_cache_clear_threshold": 0,
        "trainer_cache_occupancy_after_step": 0,
        "system_os_wired_reserve": int(DEFAULT_OS_WIRED_GIB * GIB),
    })
    return terms


def _layer_scratch_bytes(
    shapes: Shapes,
    tokens: int,
    index: int,
    rank: int,
    working_gdn_checkpoint: bool,
    grad_checkpoint: bool,
) -> tuple[int, int]:
    attention, moe = layer_phase_terms(
        shapes, tokens, rank, index, True,
        working_gdn_checkpoint=working_gdn_checkpoint,
    )
    if not grad_checkpoint:
        for values in (attention, moe):
            for name in (
                "recompute_gdn_projection_activations",
                "recompute_attention_projection_activations",
                "recompute_moe_forward_activations",
                "recompute_lora_rank_activations",
            ):
                values[name] = 0
    return sum(attention.values()), sum(moe.values())


def _peak_details(
    shapes: Shapes,
    tokens: int,
    lora_layers: int,
    grad_checkpoint: bool,
    rank: int,
    assume_working_gdn_checkpoint: bool,
) -> tuple[int, str]:
    terms = budget(
        shapes, tokens, lora_layers, grad_checkpoint=grad_checkpoint, rank=rank,
        assume_working_gdn_checkpoint=assume_working_gdn_checkpoint,
    )
    persistent_names = (
        "weights_resident", "lora_parameters", "adam_m", "adam_v",
        "checkpoint_saved_frozen_prefix_activations", "checkpoint_saved_lora_segment_input",
        "checkpoint_saved_lora_layer_inputs", "checkpoint_final_layer_output",
        "noncheckpoint_saved_frozen_activations", "noncheckpoint_saved_lora_activations",
        "batch_targets_and_completion_mask", "loss_full_sequence_hidden_gradient",
    )
    persistent = sum(terms[name] for name in persistent_names)
    loss_names = (
        "final_norm_output", "logits_chunk", "loss_logits_gradient",
        "loss_head_input_gradient",
    )
    best = (persistent + sum(terms[name] for name in loss_names),
            "backward of one completion-loss chunk")
    update_names = (
        "weights_resident", "lora_parameters", "adam_m", "adam_v",
        "batch_targets_and_completion_mask", "lora_gradients", "adam_update_temporaries",
    )
    update = sum(terms[name] for name in update_names)
    if update > best[0]:
        best = (update, "Adam update after backward activations are released")
    first = shapes.layers - lora_layers
    for index in range(first, shapes.layers):
        attn, moe = _layer_scratch_bytes(
            shapes, tokens, index, rank, assume_working_gdn_checkpoint, grad_checkpoint
        )
        candidate = persistent + terms["lora_gradients"] + max(attn, moe)
        if candidate > best[0]:
            best = (candidate, f"backward replay of layer {index}: max(attention/GDN, MoE/router)")
    return best


def peak_bytes(
    shapes: Shapes,
    tokens: int,
    lora_layers: int,
    *,
    grad_checkpoint: bool = True,
    rank: int = DEFAULT_RANK,
    assume_working_gdn_checkpoint: bool = False,
) -> int:
    """Return max loss, one-layer replay, and optimizer-update live sets."""
    return _peak_details(
        shapes, tokens, lora_layers, grad_checkpoint, rank,
        assume_working_gdn_checkpoint,
    )[0]


def block_peak_bytes(
    shapes: Shapes,
    tokens: int,
    kind: str,
    *,
    gdn_checkpoint: bool = True,
    assume_working_gdn_checkpoint: bool = False,
) -> int:
    return shapes.weights_bytes + block_incremental_bytes(
        shapes, tokens, kind, gdn_checkpoint=gdn_checkpoint,
        working_gdn_checkpoint=assume_working_gdn_checkpoint,
    )


TRAINER_REGRESSION_POINTS = (
    (256, 1, 18.60), (512, 1, 18.87), (1000, 1, 19.39),
    (2000, 1, 19.85), (3000, 1, 20.11), (256, 2, 20.22),
    (512, 2, 22.34), (1000, 2, 26.36), (512, 4, 22.32),
    (1000, 4, 26.45), (768, 3, 24.46), (900, 8, 25.66),
    (900, 16, 25.70), (768, 16, 24.40),
)
MEASURED_POINTS = (
    ("probeA", 1, 470, 470, 19.48, "allocator peak"),
    ("probe4", 4, 470, 471, 25.14, "allocator peak"),
    ("train2", 2, 600, 600, 24.46, "allocator peak"),
    ("probeI", 2, 600, 600, 22.532, "allocator peak"),
    ("probeJ", 2, 600, 600, 22.528, "allocator peak"),
    ("probeC", 1, 1200, 3000, 24.19, "allocator peak"),
    ("run3", 1, 1200, 3000, 24.18, "allocator peak"),
    ("probeE", 1, 1200, 3000, 24.177, "allocator peak"),
    ("train1", 4, 600, 600, None, "OOM at 28 GiB limit"),
    ("probeB", 2, 1200, 3000, 27.5 / 1.073741824, "reported decimal-GB footprint"),
    ("probeG", 2, 1200, 3000, 27.5 / 1.073741824, "reported decimal-GB footprint"),
    ("probeH", 2, 1200, 3000, 29.93, "system wired peak"),
    ("probeF", 2, 1200, 3000, 31.26, "process resident peak"),
)


def trainer_positions(row_tokens: int, max_seq_length: int = 10240) -> int:
    if row_tokens < 1 or max_seq_length < 2:
        raise ValueError("row_tokens must be positive and max_seq_length at least 2")
    padded_ids = min(1 + 32 * ((row_tokens + 31) // 32), max_seq_length)
    return max(1, padded_ids - 1)


def _format_gib(value: int | float) -> str:
    return f"{value / GIB:.6f}"


def print_terms(
    shapes: Shapes,
    tokens: int,
    layers: int,
    rank: int,
    checkpoint: bool,
    cache_limit_gib: float | None,
    assume_working_gdn_checkpoint: bool,
) -> None:
    terms = budget(
        shapes, tokens, layers, grad_checkpoint=checkpoint, rank=rank,
        cache_limit_gib=cache_limit_gib,
        assume_working_gdn_checkpoint=assume_working_gdn_checkpoint,
    )
    print(f"{'Term':48s}{'GiB':>14s}")
    for name, value in terms.items():
        print(f"{name:48s}{_format_gib(value):>14s}")
    peak, moment = _peak_details(
        shapes, tokens, layers, checkpoint, rank, assume_working_gdn_checkpoint
    )
    limits = allocator_limits(cache_limit_gib)
    occupancy = limits.occupancy_bytes
    current = system_wired_occupancy_bytes(peak, occupancy) / GIB
    upper = system_wired_bytes(peak, cache_limit_gib) / GIB
    print(f"Predicted active peak: {_format_gib(peak)} GiB at {moment}")
    print(
        f"Predicted system wired: {current:.3f} GiB at current cache occupancy; "
        f"{upper:.3f} GiB at cache cap (cap {limits.cache_cap_bytes / GIB:.3f} GiB, "
        f"occupancy {limits.occupancy_bytes / GIB:.3f} GiB, "
        f"OS reserve {DEFAULT_OS_WIRED_GIB:.1f} GiB, guard "
        f"{MEM_GUARD_LIMIT_GIB:.1f} GiB)"
    )
    print("Trainer cache policy: clear after eval when cache exceeds 0 bytes.")
    if limits.runtime_note:
        print(f"Allocator note: {limits.runtime_note}")


def _print_regression_points(
    shapes: Shapes, rank: int, checkpoint: bool, working_gdn_checkpoint: bool
) -> None:
    print("\nTrainer-step regression points (cache limit 0, active peak):")
    print("row T | L | model T after trainer padding | predicted GiB | measured GiB | gap")
    for row_tokens, layers, measured in TRAINER_REGRESSION_POINTS:
        tokens = trainer_positions(row_tokens)
        predicted = peak_bytes(
            shapes, tokens, layers, grad_checkpoint=checkpoint, rank=rank,
            assume_working_gdn_checkpoint=working_gdn_checkpoint,
        )
        gap = 100 * (predicted / GIB - measured) / measured
        print(
            f"{row_tokens} | {layers} | {tokens} | {predicted / GIB:.3f} | "
            f"{measured:.3f} | {gap:+.1f}%"
        )


def _print_block_points(shapes: Shapes, working_gdn_checkpoint: bool) -> None:
    rows = (
        ("GDN ckpt off T128", "gdn", 128, False, 1.141),
        ("GDN ckpt off T256", "gdn", 256, False, 2.192),
        ("GDN ckpt off T512", "gdn", 512, False, 4.292),
        ("GDN ckpt off T1000", "gdn", 1000, False, None),
        ("GDN ckpt on T128", "gdn", 128, True, 0.887),
        ("GDN ckpt on T256", "gdn", 256, True, 1.617),
        ("GDN ckpt on T512", "gdn", 512, True, 3.681),
        ("MoE T1000", "moe", 1000, False, 0.190),
        ("MoE T2000", "moe", 2000, False, 0.630),
        ("Attention T1000", "attention", 1000, False, 0.300),
        ("Attention T2000", "attention", 2000, False, 0.480),
    )
    print("\nBlock value_and_grad increment above resident weights:")
    print("point | predicted GiB | measured GiB | gap")
    for label, kind, tokens, enabled, measured in rows:
        predicted = block_incremental_bytes(
            shapes, tokens, kind, gdn_checkpoint=enabled,
            working_gdn_checkpoint=working_gdn_checkpoint,
        ) / GIB
        if measured is None:
            observed, gap = ">11.000 (guard stop)", f"<{100 * (predicted - 11.0) / 11.0:+.1f}% vs lower bound"
        else:
            observed = f"{measured:.3f}"
            gap = f"{100 * (predicted - measured) / measured:+.1f}%"
        print(f"{label} | {predicted:.3f} | {observed} | {gap}")


def _print_legacy_points(
    shapes: Shapes,
    rank: int,
    checkpoint: bool,
    cache_limit_gib: float | None,
    working_gdn_checkpoint: bool,
) -> None:
    print("\nBrief 36 measurements:")
    for name, layers, low, high, measured, metric in MEASURED_POINTS:
        low_active = peak_bytes(shapes, max(1, low - 1), layers,
                                grad_checkpoint=checkpoint, rank=rank,
                                assume_working_gdn_checkpoint=working_gdn_checkpoint)
        high_active = peak_bytes(shapes, max(1, high - 1), layers,
                                 grad_checkpoint=checkpoint, rank=rank,
                                 assume_working_gdn_checkpoint=working_gdn_checkpoint)
        if metric == "system wired peak":
            low_prediction = system_wired_prediction(low_active, cache_limit_gib)
            high_prediction = system_wired_prediction(high_active, cache_limit_gib)
            label = (
                f"{low_prediction['system_wired_current_occupancy'] / GIB:.3f}–"
                f"{low_prediction['system_wired_cache_cap_bound'] / GIB:.3f} .. "
                f"{high_prediction['system_wired_current_occupancy'] / GIB:.3f}–"
                f"{high_prediction['system_wired_cache_cap_bound'] / GIB:.3f}"
            )
        else:
            label = f"{low_active / GIB:.3f} .. {high_active / GIB:.3f}"
        if measured is None:
            measured_text, gap = "censored", "n/a"
        elif metric in {"system wired peak", "process resident peak"}:
            measured_text, gap = f"{measured:.3f} GiB", "n/a (different or bounded metric)"
        else:
            low_gap = 100 * (low_active / GIB - measured) / measured
            high_gap = 100 * (high_active / GIB - measured) / measured
            measured_text = f"{measured:.3f} GiB"
            gap = f"{low_gap:+.1f} .. {high_gap:+.1f}%"
        print(f"{name}, L={layers}, row={low}..{high} | {label} GiB | {measured_text} | {gap} | {metric}")


def print_points(
    shapes: Shapes,
    rank: int,
    checkpoint: bool,
    assume_working_gdn_checkpoint: bool,
    cache_limit_gib: float | None = None,
) -> None:
    _print_legacy_points(
        shapes, rank, checkpoint, cache_limit_gib,
        assume_working_gdn_checkpoint,
    )
    _print_regression_points(
        shapes, rank, checkpoint, assume_working_gdn_checkpoint
    )
    _print_block_points(shapes, assume_working_gdn_checkpoint)
    print("\nHeld-out trainer points:")
    held_out = ((768, 3, 24.46), (1500, 2, None))
    for row_tokens, layers, measured in held_out:
        tokens = trainer_positions(row_tokens)
        predicted = peak_bytes(
            shapes, tokens, layers, grad_checkpoint=checkpoint, rank=rank,
            assume_working_gdn_checkpoint=assume_working_gdn_checkpoint,
        ) / GIB
        result = "guard stop" if measured is None else f"{measured:.3f} GiB"
        gap = "n/a (censored stop)" if measured is None else f"{100 * (predicted - measured) / measured:+.1f}%"
        print(f"row T={row_tokens}, L={layers}, model T={tokens} | {predicted:.3f} GiB | {result} | {gap}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--tokens", type=int, default=1000)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--points", action="store_true")
    parser.add_argument("--cache-limit", type=float, default=None, metavar="GIB")
    parser.add_argument("--assume-working-gdn-checkpoint", action="store_true")
    args = parser.parse_args()
    if args.cache_limit is not None and (
        not math.isfinite(args.cache_limit) or args.cache_limit < 0
    ):
        parser.error("--cache-limit must be a finite number >= 0")
    shapes = load_shapes(args.config)
    if args.points:
        print_points(
            shapes, args.rank, args.grad_checkpoint,
            args.assume_working_gdn_checkpoint, args.cache_limit,
        )
    else:
        print_terms(
            shapes, args.tokens, args.layers, args.rank,
            args.grad_checkpoint, args.cache_limit,
            args.assume_working_gdn_checkpoint,
        )


if __name__ == "__main__":
    main()
