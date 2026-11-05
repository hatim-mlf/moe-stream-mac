#!/usr/bin/env python3
"""Lead-written tests for tools/steer_dev_eval.py: uses the contrast set's own answers, no model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import steer_dev_eval as sde  # noqa: E402


def main():
    dev = sde.load_pairs("dev")
    build = sde.load_pairs("build")
    assert 10 <= len(dev) <= 20 and len(dev) + len(build) == 50, (len(dev), len(build))
    assert all(p["split"] == "dev" for p in dev)
    try:
        sde.load_pairs("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("load_pairs must raise ValueError for an empty split")
    print("ok  load_pairs")
    good = sde.evaluate(dev, lambda m: next(p["preferred"] for p in dev if p["messages"] == m))
    bad = sde.evaluate(dev, lambda m: next(p["unpreferred"] for p in dev if p["messages"] == m))
    assert all(r["passed"] for r in good), [r["task_id"] for r in good if not r["passed"]]
    assert not any(r["passed"] for r in bad), [r["task_id"] for r in bad if r["passed"]]
    assert set(good[0]) == {"task_id", "category", "passed", "output_tail"}
    none = sde.evaluate(dev[:1], lambda m: "no fence here")
    assert none[0]["passed"] is False and none[0]["output_tail"] == "no code block"
    langs = {("c" if not (sde.ROOT / "data/steering/tests" / p["task_id"] / "test.py").exists() else "py") for p in dev}
    assert langs == {"c", "py"}, langs
    print("ok  evaluate (preferred all pass, unpreferred all fail, both languages)\nALL PASS")


if __name__ == "__main__":
    main()
