#!/usr/bin/env python3
"""Fixture-gated tests for tools/router_bias.py (Phase 1c router bias). Written by the lead before the hook.

Uses a tiny random Qwen3Next MoE block (the class Qwen3.6-35B-A3B uses), so it runs in seconds without weights.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import router_bias as rb  # noqa: E402

LAYERS, EXPERTS, TOP_K, DIM = 3, 16, 4, 32


def tiny_block(seed, norm_topk_prob=True):
    mx.random.seed(seed)
    args = SimpleNamespace(hidden_size=DIM, moe_intermediate_size=8, shared_expert_intermediate_size=8,
                           norm_topk_prob=norm_topk_prob, num_experts=EXPERTS, num_experts_per_tok=TOP_K)
    block = Qwen3NextSparseMoeBlock(args)
    mx.eval(block.parameters())
    return block


class TinyModel(nn.Module):
    """Mirrors the nesting of the real model: model.language_model.model.layers[i].mlp."""

    def __init__(self, norm_topk_prob=True):
        super().__init__()
        layers = [SimpleNamespace(mlp=tiny_block(10 + i, norm_topk_prob)) for i in range(LAYERS)]
        self.language_model = SimpleNamespace(model=SimpleNamespace(layers=layers))


def x_input(seed=0, tokens=5):
    return mx.random.normal((1, tokens, DIM), key=mx.random.key(seed))


def reference_select(block, x, bias, mode):
    """Independent reference of the biased routing: returns (inds sorted, weights matched to sorted inds)."""
    logits = np.array(block.gate(x).astype(mx.float32)).reshape(-1, EXPERTS)
    b = bias[None, :]

    def softmax(z):
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    probs = softmax(logits + b) if mode == "logit" else softmax(logits)
    chooser = softmax(logits + b)
    inds = np.argsort(-chooser, axis=-1, kind="stable")[:, :TOP_K]
    weights = np.take_along_axis(probs, inds, axis=-1)
    if block.norm_topk_prob:
        weights = weights / weights.sum(axis=-1, keepdims=True)
    order = np.argsort(inds, axis=-1)
    return np.take_along_axis(inds, order, -1), np.take_along_axis(weights, order, -1)


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"{name}: {detail}")
    print(f"ok  {name}")


def test_find_blocks():
    model = TinyModel()
    blocks = rb.find_moe_blocks(model)
    check("find_moe_blocks count", len(blocks) == LAYERS, f"got {len(blocks)}")
    check("find_moe_blocks order", all(b is l.mlp for b, l in zip(blocks, model.language_model.model.layers)))


def test_zero_bias_identity():
    model = TinyModel()
    x = x_input()
    before = [np.array(l.mlp(x)) for l in model.language_model.model.layers]
    for mode in ("select", "logit"):
        rb.install_router_bias(model, np.zeros((LAYERS, EXPERTS), np.float32), mode)
        after = [np.array(l.mlp(x)) for l in model.language_model.model.layers]
        check(f"zero bias is identity ({mode})", all(np.array_equal(a, b) for a, b in zip(before, after)))
    rb.remove_router_bias(model)


def test_routing_matches_reference():
    for norm in (True, False):
        model = TinyModel(norm)
        rng = np.random.default_rng(1)
        bias = rng.normal(0, 2.0, (LAYERS, EXPERTS)).astype(np.float32)
        x = x_input(3)
        for mode in ("select", "logit"):
            rb.install_router_bias(model, bias, mode)
            for i, layer in enumerate(model.language_model.model.layers):
                inds, weights = rb.route(layer.mlp, x)
                inds, weights = np.array(inds).reshape(-1, TOP_K), np.array(weights).reshape(-1, TOP_K)
                order = np.argsort(inds, axis=-1)
                inds, weights = np.take_along_axis(inds, order, -1), np.take_along_axis(weights, order, -1)
                ref_inds, ref_w = reference_select(layer.mlp, x, bias[i], mode)
                check(f"route inds layer {i} {mode} norm={norm}", np.array_equal(inds, ref_inds))
                check(f"route weights layer {i} {mode} norm={norm}", np.allclose(weights, ref_w, atol=1e-5),
                      f"max err {np.abs(weights - ref_w).max()}")
        rb.remove_router_bias(model)


def test_large_bias_forces_set():
    model = TinyModel()
    mask = np.zeros((LAYERS, EXPERTS), bool)
    mask[:, [1, 4, 7, 9, 12]] = True
    bias = rb.set_bias(mask, 50.0)
    check("set_bias values", bias.dtype == np.float32 and np.all(bias[mask] == 50.0) and np.all(bias[~mask] == 0))
    rb.install_router_bias(model, bias, "select")
    for layer in model.language_model.model.layers:
        inds, _ = rb.route(layer.mlp, x_input(5))
        check("large bias picks only set experts", np.all(mask[0][np.array(inds)]))
    rb.remove_router_bias(model)


def test_forward_uses_bias():
    model = TinyModel()
    block = model.language_model.model.layers[0].mlp
    x = x_input(7)
    plain = np.array(block(x))
    mask = np.zeros((LAYERS, EXPERTS), bool)
    mask[:, :TOP_K] = True
    rb.install_router_bias(model, rb.set_bias(mask, 50.0), "select")
    biased = np.array(block(x))
    check("forward output changes under strong bias", not np.allclose(plain, biased))
    rb.remove_router_bias(model)
    check("remove restores original output", np.array_equal(np.array(block(x)), plain))


def test_sets_and_validation():
    rng = np.random.default_rng(0)
    coding = rng.integers(0, 100, (LAYERS, EXPERTS)).astype(np.int64)
    other = rng.integers(0, 100, (LAYERS, EXPERTS)).astype(np.int64)
    coding[:, 3] += 10_000
    mask = rb.coding_set(coding, other, 2)
    check("coding_set shape/dtype", mask.shape == (LAYERS, EXPERTS) and mask.dtype == bool)
    check("coding_set n per layer", np.all(mask.sum(axis=1) == 2))
    check("coding_set picks over-used expert", np.all(mask[:, 3]))
    rnd = rb.random_set((LAYERS, EXPERTS), 2, seed=123)
    check("random_set n per layer", np.all(rnd.sum(axis=1) == 2))
    check("random_set deterministic", np.array_equal(rnd, rb.random_set((LAYERS, EXPERTS), 2, seed=123)))
    for bad in (lambda: rb.coding_set(coding, other[:, :-1], 2), lambda: rb.coding_set(coding, other, EXPERTS + 1),
                lambda: rb.install_router_bias(TinyModel(), np.zeros((LAYERS + 1, EXPERTS), np.float32), "select"),
                lambda: rb.install_router_bias(TinyModel(), np.zeros((LAYERS, EXPERTS), np.float32), "both")):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError for invalid input")
    print("ok  invalid inputs raise ValueError")


def test_ties_and_empty():
    zero = np.zeros((1, 6), np.int64)
    check("coding_set ties pick lower ids", np.array_equal(np.flatnonzero(rb.coding_set(zero, zero, 2)[0]), [0, 1]))
    try:
        rb.find_moe_blocks(SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace(mlp=object())])))
    except ValueError:
        print("ok  find_moe_blocks raises when no MoE block")
        return
    raise AssertionError("find_moe_blocks must raise ValueError when no block is found")


def test_patch_provider():
    model, calls = TinyModel(), []

    class FakeProvider:
        def load(self, model_path, adapter_path=None, draft_model_path=None):
            calls.append(model_path)
            return model, "tokenizer"

    mask = np.zeros((LAYERS, EXPERTS), bool)
    mask[:, 0] = True
    rb.patch_provider(FakeProvider, rb.set_bias(mask, 3.0), "logit")
    provider = FakeProvider()
    out1 = provider.load("m")
    first_bias = model.language_model.model.layers[0].mlp.router_bias
    out2 = provider.load("m")
    check("patched load returns original value", out1 == (model, "tokenizer") and out2 == out1)
    check("bias installed", first_bias is not None and model.language_model.model.layers[0].mlp.router_bias_mode == "logit")
    check("bias installed once per model", model.language_model.model.layers[0].mlp.router_bias is first_bias)
    rb.remove_router_bias(model)


if __name__ == "__main__":
    for test in (test_find_blocks, test_zero_bias_identity, test_routing_matches_reference,
                 test_large_bias_forces_set, test_forward_uses_bias, test_sets_and_validation,
                 test_ties_and_empty, test_patch_provider):
        test()
    print("ALL PASS")
