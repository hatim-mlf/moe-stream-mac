```python
import sys
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock, sum_gradients

MODES = ("select", "logit")

def find_moe_blocks(model) -> list:
    layers = getattr(model, "language_model", None).model.layers if hasattr(model, "language_model") else model.model.layers
    blocks = [layer.mlp for layer in layers if isinstance(layer.mlp, Qwen3NextSparseMoeBlock)]
    if not blocks:
        raise ValueError("no Qwen3NextSparseMoeBlock found")
    return blocks

def route(block, x) -> tuple:
    logits = block.gate(x)
    b = getattr(block, "router_bias", None)
    if b is not None:
        chooser = mx.softmax(logits + b, axis=-1, precise=True)
    else:
        chooser = mx.softmax(logits, axis=-1, precise=True)
    mode = getattr(block, "router_bias_mode", "select")
    if mode == "select":
        probs = mx.softmax(logits, axis=-1, precise=True)
    else:
        probs = chooser
    k = block.top_k
    inds = mx.argpartition(chooser, kth=-k, axis=-1)[..., -k:]
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(probs, inds, axis=-1)
    if block.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    return inds, scores

def biased_call(self, x):
    if self.sharding_group is not None:
        x = sum_gradients(self.sharding_group)(x)
    inds, scores = route(self, x)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    y = y + shared_y
    if self.sharding_group is not None:
        y = mx.distributed.all_sum(y, group=self.sharding_group)
    return y

def install_router_bias(model, bias: np.ndarray, mode: str) -> None:
    if mode not in MODES:
        raise ValueError(f"Invalid mode {mode}")
    blocks = find_moe_blocks(model)
    if bias.shape[0] != len(blocks):
        raise ValueError(f"Bias rows {bias.shape[0]} != blocks {len(blocks)}")
    num_experts = blocks[0].num_experts
    if bias.shape[1] != num_experts:
        raise ValueError(f"Bias cols {bias.shape[1]} != experts {num_experts}")
    for i, block in enumerate(blocks):
        row = bias[i]
        if np.all(row == 0.0):
            block.router_bias = None
        else:
            block.router_bias = mx.array(row, dtype=mx.float32)
        block.router_bias_mode = mode
    Qwen3NextSparseMoeBlock.__call__ = biased_call

def remove_router_bias(model) -> None:
    blocks = find_moe_blocks(model)
    for block in blocks:
        block.router_bias = None
    Qwen3NextSparseMoeBlock.__call__ = Qwen3NextSparseMoeBlock.__bases__[0].__call__

def set_bias(mask: np.ndarray, strength: float) -> np.ndarray:
    return np.where(mask, strength, 0.0).astype(np.float32)

def coding_set(coding_counts: np.ndarray, other_counts: np.ndarray, n_per_layer: int) -> np.ndarray:
    if coding_counts.shape != other_counts.shape:
        raise ValueError("Shapes must match")
    layers, experts = coding_counts.shape
    if not (0 <= n_per_layer <= experts):
        raise ValueError("n_per_layer out of range")
    mask = np.zeros((layers, experts), dtype=bool)
    for i in range(layers):
        c = coding_counts[i]
        o = other_counts[i]
        C = c.sum()
        O = o.sum()
        E = experts
        log_ratios = np.log((c + 1) / (C + E)) - np.log((o + 1) / (O + E))
        indices = np.argsort(-log_ratios, kind="stable")[:n_per_layer]
        mask[i, indices] = True
    return mask

def random_set(shape: tuple, n_per_layer: int, seed: int) -> np.ndarray:
    layers, experts = shape
    mask = np.zeros((layers, experts), dtype=bool)
    rng = np.random.default_rng(seed)
    for i in range(layers):
        indices = rng.choice(experts, size=n_per_layer, replace=False)
        mask[i, indices] = True
    return mask

def patch_provider(provider_cls, bias: np.ndarray, mode: str) -> None:
    original_load = provider_cls.load
    patched_models = set()
    def patched_load(self, *a, **kw):
        result = original_load(self, *a, **kw)
        model = result[0]
        if id(model) not in patched_models:
            install_router_bias(model, bias, mode)
            patched_models.add(id(model))
        return result
    provider_cls.load = patched_load

def main(argv=None) -> int:
    import argparse
    import mlx_lm.server
    
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--bias", required=True)
    parser.add_argument("--mode", choices=MODES, default="select")
    args, rest = parser.parse_known_args(argv)
    
    bias = np.load(args.bias)
    patch_provider(mlx_lm.server.ModelProvider, bias, args.mode)
    sys.argv = ["mlx_lm.server"] + rest
    mlx_lm.server.main()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```