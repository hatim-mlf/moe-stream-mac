"""Phase 1c dev sweep: teacher-forced NLL under router bias (coding set vs random set). One model load."""
import json, sys, time
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx_lm
sys.path.insert(0, "tools")
import route_nll as rn, router_bias as rb

R = Path(sys.argv[1]); MAXLEN = 2048
mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
model, tok = mlx_lm.load(str(Path.home() / "models/Qwen3.6-35B-A3B-4bit"))
sets = {"code": {"rows": [], "src": ["data/finetune/train.jsonl", "data/finetune/valid.jsonl"]},
        "other": {"rows": [], "src": [str(R / "other_greedy.jsonl")]},
        "code_greedy": {"rows": [], "src": [str(R / "coding_greedy.jsonl")]}}
for s in sets.values():
    for f in s["src"]:
        for line in open(f):
            p, full = rn.split_ids(tok, json.loads(line)["messages"])
            if len(full) <= MAXLEN:
                s["rows"].append((p, full))

LM = model.language_model

def lean_nll(prompt, full, chunk=256):
    """Same quantity as route_nll.completion_nll, but logits only on answer positions, in chunks."""
    hidden = LM.model(mx.array([full[:-1]]))[0, len(prompt) - 1:]
    answer = mx.array(full[len(prompt):])
    total = 0.0
    for i in range(0, answer.shape[0], chunk):
        h = hidden[i:i + chunk]
        logits = (LM.lm_head(h) if hasattr(LM, "lm_head") else LM.model.embed_tokens.as_linear(h)).astype(mx.float32)
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        total -= mx.take_along_axis(lp, answer[i:i + chunk, None], axis=-1).sum().item()
    return total, answer.shape[0]

def score():
    out = {}
    for name, s in sets.items():
        tot = n = 0.0
        for p, full in s["rows"]:
            a, b = lean_nll(p, full); tot += a; n += b
            mx.clear_cache()
        out[name] = round(tot / n, 5)
    return out

coding = np.load(R / "coding_set_n16.npy"); rand = rb.random_set(coding.shape, 16, 20261105)
p0, f0 = sets["code"]["rows"][0]
ref, lean = rn.completion_nll(model, p0, f0)[0], lean_nll(p0, f0)[0]
print("check lean vs reference nll", ref, lean, flush=True); assert abs(ref - lean) <= 1e-3 * abs(ref) + 1e-3
results = []
t = time.time(); results.append({"arm": "base", "b": 0, **score()}); print(results[-1], round(time.time() - t), "s", flush=True)
for b in (0.5, 1.0, 2.0, 4.0):
    for arm, mask in (("coding16", coding), ("random16", rand)):
        rb.install_router_bias(model, rb.set_bias(mask, b), "select")
        results.append({"arm": arm, "b": b, **score()}); print(results[-1], flush=True)
        rb.remove_router_bias(model)
json.dump({"maxlen": MAXLEN, "n_rows": {k: len(v["rows"]) for k, v in sets.items()}, "mode": "select",
           "results": results}, open(R / "sweep_nll.json", "w"), indent=1)
