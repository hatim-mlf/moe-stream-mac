"""Check a converted PEFT adapter on the real 4-bit model (brief 51, step 4).

  .venv/bin/python tools/check_peft_adapter.py MODEL_DIR MLX_ADAPTER_DIR PEFT_DIR OUT_JSON

1. Coverage: the LoRA params mlx-lm creates equal the keys in the adapter file (load_weights(strict=False) would
   leave an unfilled module at B = 0 without an error), and layer 0 is reached.
2. Numerics: for one mlp and one self_attn module, LoRALinear(x) - base(x) equals scale * B A x from the original
   PEFT tensors (fp32 inputs).
3. Generation: 5 short coding prompts, greedy, thinking off, 64 tokens, base then adapter (same loaded weights).
"""
import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tuner.utils import load_adapters
from safetensors import safe_open

MAX_TOKENS = 64
NUMERIC_MODULES = ("0.mlp.down_proj", "3.self_attn.q_proj", "63.mlp.up_proj")
PEFT_PREFIX = "base_model.model.model.language_model.layers"
REL_TOL = 1e-4
PROMPTS = [
    "Write a Python function `is_palindrome(s: str) -> bool` that ignores case and non-alphanumeric characters.",
    "Fix the bug: `def mean(xs): return sum(xs) / len(xs) - 1`. Reply with the corrected function only.",
    "In C, write `int count_bits(unsigned x)` that returns the number of set bits.",
    "Explain in two sentences what a Python generator is, then show a 3-line example.",
    "Write a bash one-liner that prints the 5 largest files under the current directory.",
]


def generate_all(model, tokenizer) -> list[dict]:
    sampler = make_sampler(temp=0.0)
    out = []
    for prompt in PROMPTS:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                             enable_thinking=False, tokenize=False)
        tokens, pieces, last = [], [], None
        for resp in stream_generate(model, tokenizer, text, max_tokens=MAX_TOKENS, sampler=sampler):
            tokens.append(resp.token)
            pieces.append(resp.text)
            last = resp
        out.append({"prompt": prompt, "tokens": tokens, "text": "".join(pieces),
                    "prompt_tps": last.prompt_tps, "decode_tps": last.generation_tps,
                    "peak_gb": last.peak_memory})
    return out


def check_numerics(model, peft_dir: Path, scale: float) -> list[dict]:
    peft = safe_open(str(peft_dir / "adapter_model.safetensors"), "numpy")
    rng = np.random.default_rng(0)
    results = []
    for name in NUMERIC_MODULES:
        layer, path = name.split(".", 1)
        module = dict(model.language_model.model.layers[int(layer)].named_modules())[path]
        a = peft.get_tensor(f"{PEFT_PREFIX}.{layer}.{path}.lora_A.weight")
        b = peft.get_tensor(f"{PEFT_PREFIX}.{layer}.{path}.lora_B.weight")
        x = rng.standard_normal((4, a.shape[1])).astype(np.float32)
        got = np.array(module(mx.array(x)) - module.linear(mx.array(x)))
        want = scale * (x @ a.T) @ b.T
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        results.append({"module": name, "rel_err": rel, "ok": rel < REL_TOL})
    return results


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    model_dir, adapter_dir, peft_dir, out_json = (Path(p).expanduser() for p in sys.argv[1:5])
    start = time.time()
    model, tokenizer = load(str(model_dir))
    base = generate_all(model, tokenizer)

    model = load_adapters(model, str(adapter_dir))
    model.eval()
    file_keys = set(mx.load(str(adapter_dir / "adapters.safetensors")))
    lora_params = {k for k, _ in tree_flatten(model.parameters()) if k.rsplit(".", 1)[-1] in ("lora_a", "lora_b")}
    coverage = {"file_tensors": len(file_keys), "model_lora_params": len(lora_params),
                "equal": lora_params == file_keys,
                "layer0_has_lora": any(".layers.0." in k for k in lora_params)}
    scale = json.loads((adapter_dir / "adapter_config.json").read_text())["lora_parameters"]["scale"]
    numerics = check_numerics(model, peft_dir, scale)
    adapted = generate_all(model, tokenizer)

    gens = [{"prompt": b["prompt"], "base": b["text"], "pi": p["text"],
             "first_divergent_token": first_divergence(b["tokens"], p["tokens"]),
             "base_decode_tps": b["decode_tps"], "pi_decode_tps": p["decode_tps"],
             "pi_prompt_tps": p["prompt_tps"], "pi_peak_gb": p["peak_gb"]} for b, p in zip(base, adapted)]
    report = {"coverage": coverage, "numerics": numerics, "generations": gens,
              "all_differ": all(g["first_divergent_token"] is not None for g in gens),
              "wall_s": round(time.time() - start, 1)}
    Path(out_json).write_text(json.dumps(report, indent=2))
    print(json.dumps({"coverage": coverage, "numerics": numerics, "all_differ": report["all_differ"]}))


if __name__ == "__main__":
    main()
