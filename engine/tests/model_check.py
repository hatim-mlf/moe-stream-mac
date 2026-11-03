"""Check engine output against MLX, loading ONE layer at a time (fits a busy 32 GB Mac).

  python tests/model_check.py tokenize <model dir> <ids.i32> [prompt]
  python tests/model_check.py check    <model dir> <engine_out.bin>
check recomputes MLX logits for the engine's whole sequence (prompt + its greedy tokens) and requires
(1) the same argmax at every position, so the engine's generation is exactly what MLX would generate, and
(2) a small relative L2 error on the logits.
"""
import gc
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs
from mlx.utils import tree_map
from mlx_lm.utils import load_tokenizer

REL_TOL = 1e-3   # MLX runs float32 here (layer-at-a-time keeps memory low)
PROMPT = "The capital of Morocco is Rabat. The capital of France is"


def tensors(model_dir: Path, prefix: str) -> dict:
    out = {}
    for shard in sorted(model_dir.glob("*.safetensors")):
        out |= {k[len(prefix):]: v for k, v in mx.load(str(shard)).items() if k.startswith(prefix)}
    return out


def to_f32(module):
    module.update(tree_map(lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p, module.parameters()))
    return module


def build(module, weights: dict, qcfg: dict, full_prefix: str):
    def pred(path, m):
        if not hasattr(m, "to_quantized") or f"{path}.scales" not in weights:
            return False
        return qcfg.get(f"{full_prefix}{path}", True)
    nn.quantize(module, group_size=qcfg["group_size"], bits=qcfg["bits"], class_predicate=pred)
    module.load_weights(list(weights.items()), strict=True)
    module.eval()
    return to_f32(module)


def mlx_logits(model_dir: Path, ids: list[int]) -> np.ndarray:
    cfg = json.loads((model_dir / "config.json").read_text())
    args = TextModelArgs.from_dict(cfg.get("text_config", cfg)); q = cfg["quantization"]
    def bits(key):  # per-tensor override, else the global setting
        o = q.get(key); return o["bits"] if isinstance(o, dict) else q["bits"]
    emb = nn.QuantizedEmbedding(args.vocab_size, args.hidden_size, group_size=q["group_size"],
                                bits=bits("language_model.model.embed_tokens"))
    emb.load_weights(list(tensors(model_dir, "language_model.model.embed_tokens.").items()), strict=True); to_f32(emb)
    h = emb(mx.array(ids)[None]).astype(mx.float32); mx.eval(h); del emb
    for i in range(args.num_hidden_layers):
        pfx = f"language_model.model.layers.{i}."
        layer = build(DecoderLayer(args, i), tensors(model_dir, pfx), q, pfx)
        h = layer(h, mask=None if layer.is_linear else "causal"); mx.eval(h)
        del layer; gc.collect(); mx.clear_cache()
        print(f"\r  mlx layer {i + 1}/{args.num_hidden_layers}", end="", flush=True)
    norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
    norm.load_weights([("weight", tensors(model_dir, "language_model.model.norm.")["weight"])]); to_f32(norm)
    head = nn.QuantizedLinear(args.hidden_size, args.vocab_size, bias=False, group_size=q["group_size"],
                              bits=bits("language_model.lm_head"))
    head.load_weights(list(tensors(model_dir, "language_model.lm_head.").items()), strict=True); to_f32(head)
    print()
    return np.array(head(norm(h))[0].astype(mx.float32))


def main() -> None:
    mode, model_dir = sys.argv[1], Path(sys.argv[2]).expanduser()
    tok = load_tokenizer(model_dir)
    if mode == "tokenize":
        ids = tok.encode(sys.argv[4] if len(sys.argv) > 4 else PROMPT)
        np.array(ids, "<i4").tofile(sys.argv[3]); print(f"{len(ids)} prompt tokens -> {sys.argv[3]}"); return
    raw = open(sys.argv[3], "rb").read()
    np_, ng, V = np.frombuffer(raw[:12], "<i4"); n = np_ + ng
    ids = np.frombuffer(raw, "<i4", n, 12).tolist()
    eng = np.frombuffer(raw, "<f4", (n - 1) * V, 12 + 4 * n).reshape(n - 1, V)
    ref = mlx_logits(model_dir, ids[:-1])
    failed = False
    for t in range(n - 1):
        rel = np.linalg.norm(eng[t] - ref[t]) / np.linalg.norm(ref[t])
        same = int(eng[t].argmax()) == int(ref[t].argmax())
        ok = same and rel < REL_TOL; failed |= not ok
        tag = "gen " if t >= np_ - 1 else "prompt"
        print(f"{'PASS' if ok else 'FAIL'}  {tag} pos {t:2d}: rel L2 {rel:.2e}, argmax {'same' if same else 'DIFFERENT'}")
    print("engine generated:", repr(tok.decode(ids[np_:])))
    print("WHOLE MODEL MATCHES MLX (engine generation = MLX greedy)" if not failed else "MODEL CHECK FAILED")
    sys.exit(failed)


if __name__ == "__main__":
    main()
