# Task: write tools/router_bias.py (Python 3.14, numpy + mlx only)

Add a per-layer bias to the MoE router of a loaded mlx_lm Qwen3.6 model, to push routing toward a chosen set of
experts. The MoE block class is `mlx_lm.models.qwen3_next.Qwen3NextSparseMoeBlock`. Its original forward is:

```python
def __call__(self, x):
    if self.sharding_group is not None:
        x = sum_gradients(self.sharding_group)(x)
    gates = self.gate(x)
    gates = mx.softmax(gates, axis=-1, precise=True)
    k = self.top_k
    inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
    inds = mx.stop_gradient(inds)
    scores = mx.take_along_axis(gates, inds, axis=-1)
    if self.norm_topk_prob:
        scores = scores / scores.sum(axis=-1, keepdims=True)
    y = self.switch_mlp(x, inds)
    y = (y * scores[..., None]).sum(axis=-2)
    shared_y = self.shared_expert(x)
    shared_y = mx.sigmoid(self.shared_expert_gate(x)) * shared_y
    y = y + shared_y
    if self.sharding_group is not None:
        y = mx.distributed.all_sum(y, group=self.sharding_group)
    return y
```

Write exactly these module-level names:

```python
MODES = ("select", "logit")

def find_moe_blocks(model) -> list:
    """Return model.language_model.model.layers[i].mlp for every i whose mlp is a Qwen3NextSparseMoeBlock, in layer
    order. If model has no language_model attribute use model.model.layers. Raise ValueError if none are found."""

def route(block, x) -> tuple:
    """Biased routing for one block: (inds, scores), both mx.array with last dim block.top_k.
    logits = block.gate(x); b = getattr(block, "router_bias", None) (an mx.array [num_experts] float32, or None).
    chooser = softmax(logits + b) if b is not None else softmax(logits)   (softmax with axis=-1, precise=True)
    mode "logit": probs = chooser.  mode "select" (block.router_bias_mode): probs = softmax(logits) (unbiased).
    inds = argpartition(chooser, kth=-k)[..., -k:] (stop_gradient); scores = take_along_axis(probs, inds);
    renormalise scores over the k only if block.norm_topk_prob. With b None this must be op-for-op the original."""

def biased_call(self, x):
    """Same as the original forward above, but inds/scores come from route(self, x)."""

def install_router_bias(model, bias: np.ndarray, mode: str) -> None:
    """bias: float array [num_blocks, num_experts]. Raise ValueError if mode not in MODES or the shape does not match
    (len(find_moe_blocks(model)), block.num_experts). For each block i: set block.router_bias = mx.array(bias[i],
    dtype=mx.float32), block.router_bias_mode = mode. Then patch the CLASS once:
    Qwen3NextSparseMoeBlock.__call__ = biased_call, keeping the original in a module-level variable _ORIGINAL_CALL
    (do not overwrite _ORIGINAL_CALL if already patched). An all-zero row sets router_bias = None for that block so
    zero bias is bit-identical to the original."""

def remove_router_bias(model) -> None:
    """Set router_bias = None on every block and restore the original class __call__."""

def set_bias(mask: np.ndarray, strength: float) -> np.ndarray:
    """float32 array with `strength` where mask is True and 0.0 elsewhere."""

def coding_set(coding_counts: np.ndarray, other_counts: np.ndarray, n_per_layer: int) -> np.ndarray:
    """Bool mask [layers, experts]: per layer, the n_per_layer experts with the highest log-ratio
    log((c + 1) / (C + E)) - log((o + 1) / (O + E)), where c, o are counts, C, O the layer totals, E the number of
    experts. Ties: lower expert id first (stable sort). Raise ValueError if shapes differ or
    not 0 <= n_per_layer <= experts."""

def random_set(shape: tuple, n_per_layer: int, seed: int) -> np.ndarray:
    """Bool mask: per layer, n_per_layer distinct experts chosen with np.random.default_rng(seed).choice(...,
    replace=False), drawing the layers in order with that one generator."""
```

Also add a command-line entry point that serves the biased model through the stock MLX server:

```
.venv/bin/python tools/router_bias.py serve --bias BIAS.npy --mode select [any mlx_lm.server args, e.g. --model M --port 8080]
```
It loads BIAS.npy, wraps `mlx_lm.server.ModelProvider.load` so that after the original load returns it calls
install_router_bias(self.model, bias, mode) once per newly loaded model, then sets sys.argv to the remaining args
and calls mlx_lm.server.main().

Rules: import numpy, mlx and mlx_lm only. Each function under 30 lines. Return only the complete file in one
fenced python block.
