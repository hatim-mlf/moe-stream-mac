"""How well does layer L+1's router, applied to layer L's MoE input, predict layer L+1's experts?

  python engine/hybrid/predict_study.py <model dir> [tokens]

For every generated token and layer L, compares the experts layer L+1 actually picks (top-k of its router on
its own input) with the top-n predicted early by running layer L+1's router on layer L's MoE input.
Reports recall = |predicted ∩ actual| / k for n = k, 1.5k, 2k. High recall means the reads for layer L+1 can
start while the GPU is still busy with layer L.
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

sys.path.insert(0, str(Path(__file__).parent))
from hybrid import HybridModel  # noqa: E402

model_dir = Path(sys.argv[1]).expanduser()
n_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 60
hm = HybridModel(model_dir, cache_gb=0)
blocks = [layer.mlp for layer in hm.model.language_model.model.layers]
block_index = {id(b): i for i, b in enumerate(blocks)}   # modules compare by contents, so index by identity
k = blocks[0].top_k
SIZES = (k, int(1.5 * k), 2 * k)
predicted: dict = {}             # layer -> {n: predicted ids} for the CURRENT token
hits = {n: np.zeros(len(blocks)) for n in SIZES}
counts = np.zeros(len(blocks))
original_call = Qwen3NextSparseMoeBlock.__call__


def topk(logits: mx.array, n: int) -> set:
    return set(np.array(mx.argpartition(logits.reshape(-1), kth=-n)[-n:]).tolist())


def instrumented(self, x):
    layer = block_index[id(self)]
    actual = topk(self.gate(x), k)
    if layer in predicted:                                   # score the prediction made one layer earlier
        for n in SIZES:
            hits[n][layer] += len(predicted[layer][n] & actual) / k
        counts[layer] += 1
    if layer + 1 < len(blocks):                              # predict the next layer from THIS layer's input
        nxt = blocks[layer + 1].gate(x)
        predicted[layer + 1] = {n: topk(nxt, n) for n in SIZES}
    return original_call(self, x)


Qwen3NextSparseMoeBlock.__call__ = instrumented
prompt = hm.tokenizer.apply_chat_template([{"role": "user", "content": "Explain how a lighthouse works, in detail."}],
                                          tokenize=False, add_generation_prompt=True, enable_thinking=False)
logits = hm.forward(hm.tokenizer.encode(prompt))
for n in SIZES:
    hits[n][:] = 0
counts[:] = 0
t0 = time.perf_counter()
for _ in range(n_tokens):
    predicted.clear()
    logits = hm.forward([int(mx.argmax(logits))])
print(f"{n_tokens} tokens in {time.perf_counter() - t0:.0f}s; recall of next-layer expert prediction (k = {k}):")
valid = counts > 0
for n in SIZES:
    per_layer = hits[n][valid] / counts[valid]
    print(f"  predict {n:2d} experts: mean recall {per_layer.mean():.0%} | worst layer {per_layer.min():.0%} | "
          f"layers >= 75%: {(per_layer >= 0.75).sum()}/{valid.sum()}")
