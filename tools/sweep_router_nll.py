#!/usr/bin/env python3
"""Phase 1c dev sweep: teacher-forced NLL under router bias (coding set vs random set). One model load."""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx_lm
import numpy as np

import mem_guard

sys.path.insert(0, "tools")
import route_nll as rn
import router_bias as rb

MAXLEN = 2048
MODEL_PATH = Path.home() / "models/Qwen3.6-35B-A3B-4bit"
MAX_NLL_CHUNK = 256
RANDOM_SET_SEED = 20261105
SWEEP_BIASES = (0.5, 1.0, 2.0, 4.0)


def main(argv: list[str] | None = None) -> int:
    _total_memory = mem_guard.total_memory_bytes()
    _headroom = mem_guard.headroom_bytes_from_env(os.environ)
    _limit = _total_memory - _headroom
    print(
        f"mem_guard: limit {_limit / 2**30:.2f} GiB "
        f"(total {_total_memory / 2**30:.2f}, headroom {_headroom / 2**30:.2f})",
        flush=True,
    )
    mem_guard.start(total_bytes=_total_memory, headroom_bytes=_headroom)

    args = sys.argv[1:] if argv is None else argv
    result_dir = Path(args[0])
    mx.set_wired_limit(mx.metal.device_info()["max_recommended_working_set_size"])
    model, tokenizer = mlx_lm.load(str(MODEL_PATH))
    sets: dict[str, dict[str, Any]] = {
        "code": {"rows": [], "src": ["data/finetune/train.jsonl", "data/finetune/valid.jsonl"]},
        "other": {"rows": [], "src": [str(result_dir / "other_greedy.jsonl")]},
        "code_greedy": {"rows": [], "src": [str(result_dir / "coding_greedy.jsonl")]},
    }
    for dataset in sets.values():
        for input_path in dataset["src"]:
            with open(input_path) as source:
                for line in source:
                    prompt, full = rn.split_ids(tokenizer, json.loads(line)["messages"])
                    if len(full) <= MAXLEN:
                        dataset["rows"].append((prompt, full))

    language_model = model.language_model

    def lean_nll(
        prompt: list[int], full: list[int], chunk: int = MAX_NLL_CHUNK
    ) -> tuple[float, int]:
        """Score answer positions in chunks, matching route_nll.completion_nll."""
        hidden = language_model.model(mx.array([full[:-1]]))[0, len(prompt) - 1 :]
        answer = mx.array(full[len(prompt) :])
        total = 0.0
        for index in range(0, answer.shape[0], chunk):
            hidden_chunk = hidden[index : index + chunk]
            logits = (
                language_model.lm_head(hidden_chunk)
                if hasattr(language_model, "lm_head")
                else language_model.model.embed_tokens.as_linear(hidden_chunk)
            ).astype(mx.float32)
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            total -= mx.take_along_axis(
                log_probs, answer[index : index + chunk, None], axis=-1
            ).sum().item()
        return total, answer.shape[0]

    def score() -> dict[str, float]:
        output: dict[str, float] = {}
        for name, dataset in sets.items():
            total_nll = 0.0
            total_tokens = 0.0
            for prompt, full in dataset["rows"]:
                nll, tokens = lean_nll(prompt, full)
                total_nll += nll
                total_tokens += tokens
                mx.clear_cache()
            output[name] = round(total_nll / total_tokens, 5)
        return output

    coding = np.load(result_dir / "coding_set_n16.npy")
    random = rb.random_set(coding.shape, 16, RANDOM_SET_SEED)
    first_prompt, first_full = sets["code"]["rows"][0]
    reference = rn.completion_nll(model, first_prompt, first_full)[0]
    lean = lean_nll(first_prompt, first_full)[0]
    print("check lean vs reference nll", reference, lean, flush=True)
    assert abs(reference - lean) <= 1e-3 * abs(reference) + 1e-3

    results: list[dict[str, Any]] = []
    started = time.time()
    results.append({"arm": "base", "b": 0, **score()})
    print(results[-1], round(time.time() - started), "s", flush=True)
    for bias_scale in SWEEP_BIASES:
        for arm, mask in (("coding16", coding), ("random16", random)):
            rb.install_router_bias(model, rb.set_bias(mask, bias_scale), "select")
            results.append({"arm": arm, "b": bias_scale, **score()})
            print(results[-1], flush=True)
            rb.remove_router_bias(model)

    output_path = result_dir / "sweep_nll.json"
    output_path.write_text(
        json.dumps(
            {
                "maxlen": MAXLEN,
                "n_rows": {key: len(value["rows"]) for key, value in sets.items()},
                "mode": "select",
                "results": results,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
