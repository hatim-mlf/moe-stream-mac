"""Source-shaped allocation formulas used by :mod:`mem_budget`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mem_budget import Shapes


GIB = 1 << 30
DEFAULT_RANK = 8
DEFAULT_GDN_CHUNK = 64
DEFAULT_LOSS_CHUNK = 512
F32_BYTES = 4
I32_BYTES = 4
BOOL_BYTES = 1
GDN_VJP_STATE_ARRAYS = 4
DEFAULT_MAX_RECOMMENDED_WORKING_SET_GIB = 26.8
DEFAULT_DEVICE_MEMORY_GIB = 32.0
# Measured max of (system wired - MLX active) over the trainer sweep and deep-LoRA runs (2.4-4.0 GiB,
# grows with L; NOTES 2026-11-05). Conservative: the largest observed gap.
DEFAULT_OS_WIRED_GIB = 4.0
MEM_GUARD_LIMIT_GIB = 29.5
PREFLIGHT_MARGIN_GIB = 1.0


def quantized_payload(elements: int, bits: int, group: int, dtype_bytes: int) -> int:
    packed = math.ceil(elements * bits / 8)
    metadata = math.ceil(elements / group) * 2 * dtype_bytes
    return packed + metadata


def lora_dimensions(
    shapes: Shapes, layer_type: str, branch: str | None = None
) -> tuple[tuple[int, int], ...]:
    hidden = shapes.hidden
    shared = shapes.shared_intermediate
    shared_dims = ((hidden, shared), (hidden, shared), (shared, hidden), (hidden, 1))
    if layer_type == "full_attention":
        q = 2 * shapes.attention_heads * shapes.head_dim
        kv = shapes.kv_heads * shapes.head_dim
        attention = (
            (hidden, q), (hidden, kv), (hidden, kv),
            (shapes.attention_heads * shapes.head_dim, hidden),
        )
    else:
        key, value = shapes.gdn_key_width, shapes.gdn_value_width
        attention = (
            (hidden, 2 * key + value), (hidden, value),
            (hidden, shapes.gdn_value_heads),
            (hidden, shapes.gdn_value_heads), (value, hidden),
        )
    if branch == "attention":
        return attention
    if branch == "moe":
        return shared_dims
    if branch is not None:
        raise ValueError(f"unknown LoRA branch: {branch}")
    return attention + shared_dims


def lora_elements(shapes: Shapes, lora_layers: int, rank: int) -> int:
    first = shapes.layers - lora_layers
    return sum(
        rank * (input_dim + output_dim)
        for kind in shapes.layer_types[first:]
        for input_dim, output_dim in lora_dimensions(shapes, kind)
    )


def moe_forward_bytes(shapes: Shapes, tokens: int) -> int:
    batch, hidden = shapes.batch_size, shapes.hidden
    route = batch * tokens * shapes.top_k
    routed = route * (2 * hidden + 3 * shapes.moe_intermediate) * shapes.dtype_bytes
    shared = batch * tokens * (
        3 * shapes.shared_intermediate + 2 * hidden + 1
    ) * shapes.dtype_bytes
    router = batch * tokens * (shapes.experts + shapes.top_k) * shapes.dtype_bytes
    return routed + shared + router


def moe_score_reduction_bytes(shapes: Shapes, tokens: int) -> int:
    rows = shapes.batch_size * tokens
    routed = rows * shapes.top_k * shapes.hidden
    reduced = rows * shapes.hidden
    return (routed + reduced) * shapes.dtype_bytes


def retained_topk_indices_bytes(shapes: Shapes, tokens: int) -> int:
    return shapes.batch_size * tokens * shapes.top_k * shapes.index_bytes


def gdn_projection_bytes(shapes: Shapes, tokens: int) -> int:
    hidden, key = shapes.hidden, shapes.gdn_key_width
    value, conv = shapes.gdn_value_width, shapes.gdn_conv_width
    heads, key_dim = shapes.gdn_value_heads, shapes.gdn_key_dim
    elems = 2 * hidden + (2 * key + value) + value + 2 * heads + 2 * conv
    elems += 2 * heads * key_dim + 3 * value + hidden
    return shapes.batch_size * tokens * (
        elems * shapes.dtype_bytes + 2 * heads * F32_BYTES
    )


def attention_projection_bytes(shapes: Shapes, tokens: int) -> int:
    hidden = shapes.hidden
    q = 2 * shapes.attention_heads * shapes.head_dim
    kv = shapes.kv_heads * shapes.head_dim
    attended = shapes.attention_heads * shapes.head_dim
    elements = 2 * hidden + q + 4 * kv + 2 * attended + hidden
    return shapes.batch_size * tokens * elements * shapes.dtype_bytes


def switch_index_bytes(shapes: Shapes, tokens: int) -> int:
    routed = shapes.batch_size * tokens * shapes.top_k
    # _gather_sort materializes the sorted route IDs, order, and inverse order.
    return 3 * routed * shapes.index_bytes


def switch_gather_vjp_bytes(shapes: Shapes, tokens: int) -> int:
    route = shapes.batch_size * tokens * shapes.top_k
    return route * (2 * shapes.hidden + shapes.moe_intermediate) * shapes.dtype_bytes


def switch_scatter_vjp_bytes(shapes: Shapes, tokens: int) -> int:
    rows = shapes.batch_size * tokens
    routed = rows * shapes.top_k * shapes.hidden
    return (routed + rows * shapes.hidden) * shapes.dtype_bytes


def quantized_attention_vjp_bytes(shapes: Shapes, tokens: int, kind: str) -> int:
    if kind == "linear_attention":
        width = 8 * shapes.hidden + shapes.gdn_value_width
    elif kind == "full_attention":
        width = 7 * shapes.hidden + shapes.attention_heads * shapes.head_dim
    else:
        raise ValueError(f"unknown layer type: {kind}")
    return shapes.batch_size * tokens * width * shapes.dtype_bytes


def quantized_moe_vjp_bytes(shapes: Shapes, tokens: int) -> int:
    rows = shapes.batch_size * tokens
    width = 3 * shapes.hidden + shapes.shared_intermediate
    return rows * width * shapes.dtype_bytes


def gdn_state_bytes(shapes: Shapes) -> int:
    return (
        shapes.batch_size * shapes.gdn_value_heads * shapes.gdn_value_dim
        * shapes.gdn_key_dim * F32_BYTES
    )


def gdn_vjp_live_bytes(shapes: Shapes, tokens: int, *, working_checkpoint: bool) -> int:
    steps = min(tokens, shapes.gdn_checkpoint_chunk) if working_checkpoint else tokens
    return GDN_VJP_STATE_ARRAYS * steps * gdn_state_bytes(shapes)


def gdn_boundary_bytes(shapes: Shapes, tokens: int, enabled: bool) -> int:
    if not enabled:
        return 0
    chunks = math.ceil(tokens / shapes.gdn_checkpoint_chunk)
    return (chunks + 1) * gdn_state_bytes(shapes)


def attention_vjp_terms(shapes: Shapes, tokens: int) -> dict[str, int]:
    heads, kv_heads, dim = shapes.attention_heads, shapes.kv_heads, shapes.head_dim
    block, dtype = min(tokens, shapes.attention_vjp_block), shapes.dtype_bytes
    q_values = shapes.batch_size * heads * tokens * dim
    kv_values = shapes.batch_size * kv_heads * tokens * dim
    return {
        "attention_sdpa_lse": shapes.batch_size * heads * tokens * F32_BYTES,
        "attention_sdpa_vjp_score_tiles": (
            3 * shapes.batch_size * heads * block * block * dtype
        ),
        "attention_sdpa_vjp_tile_gradients": (
            shapes.batch_size * heads * block * 3 * dim * dtype
        ),
        "attention_sdpa_vjp_float_accumulators": (
            q_values + 2 * kv_values
        ) * F32_BYTES,
        "attention_sdpa_vjp_output_arrays": (
            q_values + 2 * kv_values
        ) * dtype,
        "attention_sdpa_vjp_odo": shapes.batch_size * heads * tokens * F32_BYTES,
    }


def router_vjp_terms(shapes: Shapes, tokens: int) -> dict[str, int]:
    rows = shapes.batch_size * tokens
    router_values = rows * shapes.experts
    return {
        "router_softmax_vjp_arrays": (
            3 * router_values + rows
        ) * shapes.dtype_bytes,
        "router_quantized_matmul_input_gradient": (
            rows * shapes.hidden * shapes.dtype_bytes
        ),
    }


def checkpoint_boundary_terms(
    shapes: Shapes, tokens: int, lora_layers: int, grad_checkpoint: bool
) -> dict[str, int]:
    hidden = shapes.batch_size * tokens * shapes.hidden * shapes.dtype_bytes
    return {
        "checkpoint_saved_frozen_prefix_activations": 0,
        "checkpoint_saved_lora_segment_input": (
            hidden if grad_checkpoint and lora_layers else 0
        ),
        "checkpoint_saved_lora_layer_inputs": (
            max(0, lora_layers - 1) * hidden if grad_checkpoint else 0
        ),
        "checkpoint_final_layer_output": hidden if lora_layers else 0,
        "final_norm_output": hidden,
        "loss_full_sequence_hidden_gradient": hidden,
    }


def base_terms(
    shapes: Shapes, tokens: int, lora_layers: int, rank: int, grad_checkpoint: bool
) -> dict[str, int]:
    elements = lora_elements(shapes, lora_layers, rank)
    terms = {
        "lora_parameters": elements * shapes.lora_bytes,
        "lora_gradients": elements * shapes.optimizer_bytes,
        "adam_m": elements * shapes.optimizer_bytes,
        "adam_v": elements * shapes.optimizer_bytes,
        "adam_update_temporaries": 3 * elements * shapes.optimizer_bytes,
        "noncheckpoint_saved_frozen_activations": 0,
        "noncheckpoint_saved_lora_activations": 0,
        "gdn_state_bytes_per_step": gdn_state_bytes(shapes),
        "batch_targets_and_completion_mask": (
            shapes.batch_size * (tokens + 1) * shapes.index_bytes
            + shapes.batch_size * 2 * shapes.index_bytes
            + shapes.batch_size * tokens * (
                shapes.index_bytes + shapes.bool_bytes
            )
        ),
    }
    terms.update(checkpoint_boundary_terms(shapes, tokens, lora_layers, grad_checkpoint))
    if not grad_checkpoint:
        first = shapes.layers - lora_layers
        terms["noncheckpoint_saved_lora_activations"] = sum(
            layer_activation_bytes(shapes, tokens, kind, True, rank)
            for kind in shapes.layer_types[first:]
        )
    return terms


def layer_activation_bytes(
    shapes: Shapes, tokens: int, kind: str, adapted: bool, rank: int
) -> int:
    if kind == "linear_attention":
        projection = gdn_projection_bytes(shapes, tokens)
    else:
        projection = attention_projection_bytes(shapes, tokens)
    dims = lora_dimensions(shapes, kind, "attention") if adapted else ()
    lora = shapes.batch_size * len(dims) * tokens * rank * shapes.lora_bytes
    hidden = shapes.batch_size * tokens * shapes.hidden * shapes.dtype_bytes
    return hidden + projection + moe_forward_bytes(shapes, tokens) + lora


def logits_terms(shapes: Shapes, tokens: int) -> dict[str, int]:
    chunk = min(tokens, shapes.loss_chunk)
    logits = shapes.batch_size * chunk * shapes.vocab * shapes.dtype_bytes
    return {
        "logits_chunk": logits,
        "loss_logits_gradient": logits,
        "loss_head_input_gradient": (
            shapes.batch_size * chunk * shapes.hidden * shapes.dtype_bytes
        ),
    }


def _lora_phase_terms(
    shapes: Shapes, tokens: int, rank: int, dims: tuple[tuple[int, int], ...]
) -> dict[str, int]:
    rank_bytes = shapes.batch_size * tokens * rank * shapes.lora_bytes
    return {
        "recompute_lora_rank_activations": len(dims) * rank_bytes,
        "lora_matmul_vjp_tensors": (
            shapes.batch_size * tokens * sum(a + rank for a, _ in dims)
            * shapes.lora_bytes
        ),
    }


def _attention_phase_terms(
    shapes: Shapes,
    tokens: int,
    rank: int,
    kind: str,
    adapted: bool,
    working_gdn_checkpoint: bool,
) -> dict[str, int]:
    gdn = kind == "linear_attention"
    dims = lora_dimensions(shapes, kind, "attention") if adapted else ()
    result = {
        "recompute_gdn_projection_activations": gdn_projection_bytes(shapes, tokens) if gdn else 0,
        "recompute_attention_projection_activations": 0 if gdn else attention_projection_bytes(shapes, tokens),
        "quantized_matmul_input_gradients": quantized_attention_vjp_bytes(shapes, tokens, kind) if adapted else 0,
        "gdn_checkpoint_boundary_states": gdn_boundary_bytes(shapes, tokens, adapted),
        "gdn_step_vjp_state_arrays": gdn_vjp_live_bytes(
            shapes, tokens, working_checkpoint=working_gdn_checkpoint
        ) if gdn and adapted else 0,
        "gdn_step_vjp_vector_arrays": (
            shapes.batch_size * tokens * 4 * shapes.gdn_value_width * F32_BYTES
            if gdn and adapted else 0
        ),
        "gdn_mask_where_vjp": 0,
    }
    result.update(_lora_phase_terms(shapes, tokens, rank, dims))
    result.update(attention_vjp_terms(shapes, tokens) if not gdn and adapted else {
        key: 0 for key in attention_vjp_terms(shapes, tokens)
    })
    return result


def _moe_phase_terms(
    shapes: Shapes, tokens: int, rank: int, adapted: bool
) -> dict[str, int]:
    dims = lora_dimensions(shapes, "linear_attention", "moe") if adapted else ()
    result = {
        "recompute_moe_forward_activations": moe_forward_bytes(shapes, tokens),
        "moe_score_weighted_routed_output_and_reduction": moe_score_reduction_bytes(shapes, tokens),
        "retained_topk_indices": retained_topk_indices_bytes(shapes, tokens),
        "quantized_moe_shared_input_gradients": quantized_moe_vjp_bytes(shapes, tokens) if adapted else 0,
        "gather_qmm_vjp_input_gradients": switch_gather_vjp_bytes(shapes, tokens),
        "switch_gather_scatter_vjp": switch_scatter_vjp_bytes(shapes, tokens),
        "switch_index_sort_arrays": switch_index_bytes(shapes, tokens),
    }
    result.update(_lora_phase_terms(shapes, tokens, rank, dims))
    result.update(router_vjp_terms(shapes, tokens))
    return result


def layer_phase_terms(
    shapes: Shapes,
    tokens: int,
    rank: int,
    layer_index: int,
    adapted: bool,
    *,
    working_gdn_checkpoint: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Return (GDN/attention, MoE/router) scratch inventories for one layer."""
    kind = shapes.layer_types[layer_index]
    attention = _attention_phase_terms(
        shapes, tokens, rank, kind, adapted, working_gdn_checkpoint
    )
    return attention, _moe_phase_terms(shapes, tokens, rank, adapted)


def layer_term_inventory(
    shapes: Shapes,
    tokens: int,
    lora_layers: int,
    rank: int,
    *,
    working_gdn_checkpoint: bool,
) -> dict[str, int]:
    first = shapes.layers - lora_layers
    inventories = [
        layer_phase_terms(
            shapes, tokens, rank, index, True,
            working_gdn_checkpoint=working_gdn_checkpoint,
        )
        for index in range(first, shapes.layers)
    ]
    result: dict[str, int] = {}
    for phase in inventories:
        for values in phase:
            for name, size in values.items():
                result[name] = max(result.get(name, 0), size)
    if not inventories:
        empty = layer_phase_terms(shapes, tokens, rank, 0, False)
        for values in empty:
            result.update({name: 0 for name in values})
    result["recompute_lora_rank_activations"] = max(
        (phase[0]["recompute_lora_rank_activations"] + phase[1]["recompute_lora_rank_activations"]
         for phase in inventories),
        default=0,
    )
    result["lora_matmul_vjp_tensors"] = max(
        (phase[0]["lora_matmul_vjp_tensors"] + phase[1]["lora_matmul_vjp_tensors"]
         for phase in inventories),
        default=0,
    )
    return result


def budget_terms(
    shapes: Shapes,
    tokens: int,
    lora_layers: int,
    *,
    grad_checkpoint: bool,
    rank: int,
    working_gdn_checkpoint: bool,
) -> dict[str, int]:
    if tokens < 1 or rank < 1 or not 0 <= lora_layers <= shapes.layers:
        raise ValueError("tokens/rank must be positive and lora_layers in model range")
    terms = base_terms(shapes, tokens, lora_layers, rank, grad_checkpoint)
    terms.update(logits_terms(shapes, tokens))
    terms.update(layer_term_inventory(
        shapes, tokens, lora_layers, rank,
        working_gdn_checkpoint=working_gdn_checkpoint,
    ))
    return terms


def block_incremental_bytes(
    shapes: Shapes,
    tokens: int,
    kind: str,
    *,
    gdn_checkpoint: bool = True,
    working_gdn_checkpoint: bool = False,
) -> int:
    if kind == "gdn":
        state = gdn_vjp_live_bytes(
            shapes, tokens,
            working_checkpoint=working_gdn_checkpoint and gdn_checkpoint,
        )
        boundary = gdn_boundary_bytes(shapes, tokens, gdn_checkpoint)
        return (
            state + gdn_projection_bytes(shapes, tokens) + boundary
            + shapes.batch_size * tokens * 4 * shapes.gdn_value_width * F32_BYTES
            + quantized_attention_vjp_bytes(shapes, tokens, "linear_attention")
        )
    if kind == "attention":
        values = attention_vjp_terms(shapes, tokens)
        return (
            attention_projection_bytes(shapes, tokens) + sum(values.values())
            + quantized_attention_vjp_bytes(shapes, tokens, "full_attention")
        )
    if kind == "moe":
        _, values = layer_phase_terms(shapes, tokens, DEFAULT_RANK, 0, False)
        return sum(values.values()) + quantized_moe_vjp_bytes(shapes, tokens)
    raise ValueError(f"unknown block kind: {kind}")
