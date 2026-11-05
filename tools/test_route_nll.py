#!/usr/bin/env python3
"""Tests for tools/route_nll.py (lead-written). Fake models; only the 35B tokenizer files are needed."""
import json
import math
import sys
import tempfile
from pathlib import Path

import mlx.core as mx
from mlx_lm.utils import load_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_nll as rn  # noqa: E402

TOK = load_tokenizer(Path.home() / "models/Qwen3.6-35B-A3B-4bit")
MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "add two numbers"},
        {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}]


def uniform_model(vocab):
    return lambda x: mx.zeros((1, x.shape[1], vocab), dtype=mx.bfloat16)


def oracle_model(full, vocab, hit=5.0):
    """Puts logit `hit` on the true next token at every position, 0 elsewhere."""
    def call(x):
        logits = mx.zeros((1, x.shape[1], vocab))
        nxt = mx.array(full[1:x.shape[1] + 1])
        return mx.where(mx.arange(vocab)[None, None, :] == nxt[None, :, None], hit, logits)
    return call


def main():
    prompt, full = rn.split_ids(TOK, MSGS)
    assert full[:len(prompt)] == prompt and len(full) > len(prompt), "split_ids prefix"
    assert TOK.decode(full[len(prompt):]).startswith("```python"), TOK.decode(full[len(prompt):])[:40]
    for bad in (MSGS[:-1], MSGS[:-1] + [{"role": "user", "content": "x"}]):
        try:
            rn.split_ids(TOK, bad)
        except ValueError:
            continue
        raise AssertionError("split_ids must reject a last message that is not the assistant")
    print("ok  split_ids")
    vocab = 64
    small_full = [3, 5, 7, 9, 11, 13]
    nll, n = rn.completion_nll(uniform_model(vocab), small_full[:2], small_full)
    assert n == 4 and math.isclose(nll, 4 * math.log(vocab), rel_tol=1e-5), (nll, n)
    nll, n = rn.completion_nll(oracle_model(small_full, vocab), small_full[:2], small_full)
    expected = 4 * (math.log(math.exp(5.0) + vocab - 1) - 5.0)
    assert n == 4 and math.isclose(nll, expected, rel_tol=1e-4), (nll, expected)
    print("ok  completion_nll")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "d.jsonl"
        path.write_text(json.dumps({"task_id": "t1", "messages": MSGS}) + "\n" +
                        json.dumps({"task_id": "t2", "messages": MSGS}) + "\n")
        res = rn.score_file(uniform_model(len(TOK.get_vocab()) + 1024), TOK, path)
        assert [r["task_id"] for r in res["rows"]] == ["t1", "t2"]
        assert res["total_tokens"] == 2 * (len(full) - len(prompt))
        assert math.isclose(res["nll_per_token"], res["total_nll"] / res["total_tokens"])
        assert math.isclose(res["rows"][0]["nll_per_token"], math.log(len(TOK.get_vocab()) + 1024), rel_tol=1e-4)
    print("ok  score_file\nALL PASS")


if __name__ == "__main__":
    main()
