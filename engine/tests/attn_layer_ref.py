"""MLX reference for one full-attention layer (Qwen3NextAttention) on real weights, causal, T tokens.

  python tests/attn_layer_ref.py <model dir> <layer> <T> <out.bin>
Layout: int32 T, int32 hidden, then x[T*hidden] f32, y[T*hidden] f32.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.qwen3_5 import TextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextAttention

model_dir, layer, T, out = Path(sys.argv[1]).expanduser(), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = json.loads((model_dir / "config.json").read_text())
args = TextModelArgs.from_dict(cfg.get("text_config", cfg))
prefix = f"language_model.model.layers.{layer}.self_attn."
weights = {}
for shard in model_dir.glob("*.safetensors"):
    weights |= {k[len(prefix):]: v for k, v in mx.load(str(shard)).items() if k.startswith(prefix)}
assert weights, f"layer {layer} has no self_attn (not a full-attention layer?)"

module = Qwen3NextAttention(args)
q = cfg["quantization"]
nn.quantize(module, group_size=q["group_size"], bits=q["bits"],
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in weights)
module.load_weights(list(weights.items()), strict=True)
module.eval()

rng = np.random.default_rng(5)
x = (rng.normal(size=(1, T, args.hidden_size)) * 0.5).astype(np.float32)
y = module(mx.array(x), mask="causal")
with open(out, "wb") as f:
    np.array([T, args.hidden_size], "<i4").tofile(f)
    x.astype("<f4").tofile(f)
    np.array(y.astype(mx.float32)).astype("<f4").tofile(f)
print(f"attn layer {layer}: {T} tokens -> {out}  (rope dims {int(args.head_dim * args.partial_rotary_factor)}, theta {args.rope_theta})")
