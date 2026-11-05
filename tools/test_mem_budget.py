"""Small pure-Python checks for the source-derived memory inventory."""

import json
from pathlib import Path

import pytest

from mem_budget import (
    GIB,
    TRAINER_REGRESSION_POINTS,
    budget,
    default_cache_limit_bytes,
    load_shapes,
    peak_bytes,
    system_wired_bytes,
    trainer_positions,
)
from mem_preflight import tokenize_rows
from mem_budget_terms import DEFAULT_OS_WIRED_GIB


MODEL_CONFIG = Path("/Users/hatim/models/Qwen3.6-35B-A3B-4bit/config.json")


def _minimal_config() -> dict:
    return {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 16,
            "layer_types": ["linear_attention"],
            "dtype": "bfloat16",
            "num_hidden_layers": 1,
            "vocab_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 8,
            "shared_expert_intermediate_size": 8,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 4,
            "linear_value_head_dim": 4,
            "linear_conv_kernel_dim": 4,
            "tie_word_embeddings": False,
        },
        "quantization_config": {
            "bits": 4,
            "group_size": 64,
            "model.layers.0.mlp.gate": {"bits": 8, "group_size": 64},
        },
    }


def test_shape_loader_refuses_missing_dimension(tmp_path: Path) -> None:
    config = _minimal_config()
    del config["text_config"]["hidden_size"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(KeyError, match="hidden_size"):
        load_shapes(path)


def test_terms_follow_token_and_rank_formulas() -> None:
    shapes = load_shapes(MODEL_CONFIG)
    small = budget(shapes, 8, 2)
    doubled = budget(shapes, 16, 2)
    linear_terms = (
        "checkpoint_saved_lora_segment_input",
        "checkpoint_saved_lora_layer_inputs",
        "checkpoint_final_layer_output",
        "logits_chunk",
        "loss_logits_gradient",
        "loss_head_input_gradient",
        "loss_full_sequence_hidden_gradient",
        "final_norm_output",
        "recompute_moe_forward_activations",
        "moe_score_weighted_routed_output_and_reduction",
        "retained_topk_indices",
        "recompute_gdn_projection_activations",
        "recompute_attention_projection_activations",
        "recompute_lora_rank_activations",
        "lora_matmul_vjp_tensors",
        "quantized_matmul_input_gradients",
        "gather_qmm_vjp_input_gradients",
        "switch_gather_scatter_vjp",
        "gdn_step_vjp_state_arrays",
        "gdn_step_vjp_vector_arrays",
        "attention_sdpa_lse",
        "attention_sdpa_vjp_tile_gradients",
        "attention_sdpa_vjp_float_accumulators",
        "attention_sdpa_vjp_output_arrays",
        "attention_sdpa_vjp_odo",
        "router_softmax_vjp_arrays",
        "router_quantized_matmul_input_gradient",
    )
    for name in linear_terms:
        assert doubled[name] == 2 * small[name], name
    assert doubled["batch_targets_and_completion_mask"] == (
        small["batch_targets_and_completion_mask"] + 8 * (4 + 4 + 1)
    )
    assert doubled["switch_index_sort_arrays"] == (
        small["switch_index_sort_arrays"] + 3 * 8 * 8 * 4
    )
    assert doubled["gdn_checkpoint_boundary_states"] == small["gdn_checkpoint_boundary_states"]
    assert doubled["attention_sdpa_vjp_score_tiles"] == 4 * small["attention_sdpa_vjp_score_tiles"]
    assert doubled["gdn_state_bytes_per_step"] == small["gdn_state_bytes_per_step"]
    assert doubled["lora_parameters"] == small["lora_parameters"]
    assert doubled["adam_update_temporaries"] == small["adam_update_temporaries"]
    assert doubled["adam_m"] == small["adam_m"]
    assert doubled["weights_resident"] == small["weights_resident"]
    assert doubled["gdn_step_vjp_vector_arrays"] == 2 * small["gdn_step_vjp_vector_arrays"]
    assert small["gdn_step_vjp_state_arrays"] == (
        4 * 8 * small["gdn_state_bytes_per_step"]
    )


def test_working_gdn_checkpoint_flag_uses_chunk_boundaries() -> None:
    shapes = load_shapes(MODEL_CONFIG)
    terms = budget(shapes, 1000, 2, assume_working_gdn_checkpoint=True)
    assert terms["gdn_step_vjp_state_arrays"] == (
        4 * min(1000, shapes.gdn_checkpoint_chunk) * terms["gdn_state_bytes_per_step"]
    )
    assert terms["gdn_checkpoint_boundary_states"] == (
        (1000 + shapes.gdn_checkpoint_chunk - 1) // shapes.gdn_checkpoint_chunk + 1
    ) * terms["gdn_state_bytes_per_step"]


def test_trainer_regression_points_are_within_ten_percent() -> None:
    shapes = load_shapes(MODEL_CONFIG)
    errors = []
    for row_tokens, layers, measured in TRAINER_REGRESSION_POINTS:
        model_tokens = trainer_positions(row_tokens)
        predicted = peak_bytes(shapes, model_tokens, layers) / GIB
        gap = 100 * (predicted - measured) / measured
        errors.append((row_tokens, layers, gap))
        assert abs(gap) < 10, (
            f"row T={row_tokens}, L={layers}: {predicted:.3f} vs "
            f"{measured:.3f} GiB ({gap:+.2f}%)"
        )
    assert len(errors) == 14


class _TemplateTokenizer:
    def apply_chat_template(self, messages, *, tools=None, return_dict=False):
        return list(range(sum(len(message["content"]) for message in messages)))


def test_preflight_tokenizes_train_and_valid_rows_with_chat_template(
    tmp_path: Path,
) -> None:
    for split, content in (("train", "abc"), ("valid", "xy")):
        (tmp_path / f"{split}.jsonl").write_text(
            json.dumps({"messages": [{"role": "user", "content": content}]}) + "\n"
        )
    rows = tokenize_rows(tmp_path, _TemplateTokenizer())
    assert rows == (("train.jsonl:1", 3), ("valid.jsonl:1", 2))


def test_lora_memory_is_token_independent_and_rank_linear() -> None:
    shapes = load_shapes(MODEL_CONFIG)
    rank8 = budget(shapes, 1000, 2, rank=8)
    rank16 = budget(shapes, 1000, 2, rank=16)
    other_tokens = budget(shapes, 2000, 2, rank=8)
    assert rank8["lora_parameters"] == other_tokens["lora_parameters"]
    assert rank16["lora_parameters"] == 2 * rank8["lora_parameters"]
    assert rank16["lora_gradients"] == 2 * rank8["lora_gradients"]


def test_weights_match_safetensors_payload_when_present() -> None:
    if not MODEL_CONFIG.exists() or not list(MODEL_CONFIG.parent.glob("*.safetensors")):
        pytest.skip("local model safetensors are not installed")
    shapes = load_shapes(MODEL_CONFIG)
    assert shapes.weights_exact
    assert shapes.weights_payload_bytes == 20_401_929_952
    assert shapes.weights_dropped_bytes == 893_142_496
    assert shapes.weights_bytes == 19_508_787_456
    measured_active = 19_508_787_464
    assert abs(shapes.weights_bytes - measured_active) / measured_active < 0.01


def test_system_wired_capacity_tracks_cache_limit() -> None:
    active = 20 * GIB
    assert system_wired_bytes(active, 0.0) == active + int(DEFAULT_OS_WIRED_GIB * GIB)
    assert system_wired_bytes(active, 2.0) - system_wired_bytes(active, 0.0) == 2 * GIB
    assert default_cache_limit_bytes() == int(30.4 * GIB)


def test_budget_exposes_cache_bound_and_zero_cache_wired_estimate() -> None:
    shapes = load_shapes(MODEL_CONFIG)
    from mem_budget import peak_bytes

    active = peak_bytes(shapes, 1000, 2)
    assert system_wired_bytes(active, 0.0) == active + int(DEFAULT_OS_WIRED_GIB * GIB)
    assert system_wired_bytes(active) - system_wired_bytes(active, 0.0) == (
        default_cache_limit_bytes()
    )

