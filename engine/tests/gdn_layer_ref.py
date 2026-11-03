"""MLX reference for one GatedDeltaNet layer on real weights: writes inputs + outputs for T tokens.

  python tests/gdn_layer_ref.py <model dir> <layer> <T> <out.bin>
Layout: int32 T, int32 hidden, then x[T*hidden] f32, y[T*hidden] f32.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.qwen3_5 import GatedDeltaNet, TextModelArgs

model_dir, layer, T, out = Path(sys.argv[1]).expanduser(), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = json.loads((model_dir / "config.json").read_text())
args = TextModelArgs.from_dict(cfg.get("text_config", cfg))
prefix = f"language_model.model.layers.{layer}.linear_attn."
weights = {}
for shard in model_dir.glob("*.safetensors"):
    weights |= {k[len(prefix):]: v for k, v in mx.load(str(shard)).items() if k.startswith(prefix)}

module = GatedDeltaNet(args)
q = cfg["quantization"]
nn.quantize(module, group_size=q["group_size"], bits=q["bits"],
            class_predicate=lambda p, m: isinstance(m, nn.Linear) and f"{p}.scales" in weights)
module.load_weights(list(weights.items()), strict=True)
module.eval()

rng = np.random.default_rng(3)
x = (rng.normal(size=(1, T, args.hidden_size)) * 0.5).astype(np.float32)
y = module(mx.array(x).astype(mx.float32))           # no cache: runs the full sequence causally
with open(out, "wb") as f:
    np.array([T, args.hidden_size], "<i4").tofile(f)
    x.astype("<f4").tofile(f)
    np.array(y.astype(mx.float32)).astype("<f4").tofile(f)
print(f"layer {layer}: {T} tokens, hidden {args.hidden_size} -> {out}")
