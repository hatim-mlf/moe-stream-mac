"""MLX reference for one sparse MoE block on real weights. Layout: int32 T, hidden; x[T*hidden]; y[T*hidden]."""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.qwen3_5 import TextModelArgs
from mlx_lm.models.qwen3_next import Qwen3NextSparseMoeBlock

model_dir, layer, T, out = Path(sys.argv[1]).expanduser(), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
cfg = json.loads((model_dir / "config.json").read_text())
args = TextModelArgs.from_dict(cfg.get("text_config", cfg))
prefix = f"language_model.model.layers.{layer}.mlp."
weights = {}
for shard in model_dir.glob("*.safetensors"):
    weights |= {k[len(prefix):]: v for k, v in mx.load(str(shard)).items() if k.startswith(prefix)}
qcfg = cfg["quantization"]


def predicate(path, module):
    if not hasattr(module, "to_quantized") or f"{path}.scales" not in weights:
        return False
    return qcfg.get(f"language_model.model.layers.{layer}.mlp.{path}", True)  # per-layer bit overrides


module = Qwen3NextSparseMoeBlock(args)
nn.quantize(module, group_size=qcfg["group_size"], bits=qcfg["bits"], class_predicate=predicate)
module.load_weights(list(weights.items()), strict=True)
module.eval()
rng = np.random.default_rng(9)
x = (rng.normal(size=(1, T, args.hidden_size)) * 0.5).astype(np.float32)
y = module(mx.array(x))
with open(out, "wb") as f:
    np.array([T, args.hidden_size], "<i4").tofile(f)
    x.astype("<f4").tofile(f)
    np.array(y.astype(mx.float32)).astype("<f4").tofile(f)
print(f"moe layer {layer}: {T} tokens, E={args.num_experts}, top_k={args.num_experts_per_tok} -> {out}")
