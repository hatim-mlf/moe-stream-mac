#!/usr/bin/env python3
"""Tests for tools/route_prompts.py (lead-written). Needs only the 35B tokenizer files, not the weights."""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from mlx_lm.utils import load_tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
import route_prompts as rp  # noqa: E402

MODEL = Path.home() / "models/Qwen3.6-35B-A3B-4bit"


def expect_value_error(items, label):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "p.json"
        path.write_text(json.dumps(items))
        try:
            rp.load_prompts(path)
        except ValueError:
            print(f"ok  rejects {label}")
            return
    raise AssertionError(f"load_prompts accepted {label}")


def main():
    tok = load_tokenizer(MODEL)
    ids = rp.chat_ids(tok, "SYS", "hello there")
    text = tok.decode(ids)
    assert all(isinstance(i, int) for i in ids), "ids must be ints"
    assert "SYS" in text and "hello there" in text, text
    assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), f"thinking must be off: {text[-60:]!r}"
    no_sys = tok.decode(rp.chat_ids(tok, None, "hello there"))
    assert "SYS" not in no_sys
    print("ok  chat_ids")
    expect_value_error({"id": "a", "prompt": "b"}, "non-list")
    expect_value_error([{"id": "a"}], "missing prompt")
    expect_value_error([{"id": "a/b", "prompt": "x"}], "bad id")
    expect_value_error([{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}], "duplicate id")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "p.json").write_text(json.dumps([{"id": "one", "prompt": "hi"}, {"id": "two", "prompt": "bye"}]))
        (tmp / "sys.txt").write_text("SYSTEM TEXT")
        code = rp.main(["--model", str(MODEL), "--prompts", str(tmp / "p.json"), "--out", str(tmp / "out"),
                        "--system-file", str(tmp / "sys.txt")])
        assert code == 0
        got = np.fromfile(tmp / "out/one.i32", "<i4").tolist()
        assert got == rp.chat_ids(tok, "SYSTEM TEXT", "hi"), "file content must equal chat_ids"
        assert (tmp / "out/two.i32").exists()
    print("ok  main writes .i32 files\nALL PASS")


if __name__ == "__main__":
    main()
