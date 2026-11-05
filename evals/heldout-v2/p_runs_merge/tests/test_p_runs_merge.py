import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import merge_sorted_runs


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "runs",
    ", ".join(inspect.signature(merge_sorted_runs).parameters),
)
rng = np.random.default_rng(20261105)


def check(case, index, expected, got):
    if expected != got:
        raise AssertionError(
            f"case {case} index {index}: expected {expected!r}, got {got!r}"
        )


def error(case, typ, fn):
    try:
        got = fn()
    except Exception as exc:
        check(case, 0, typ, type(exc))
        return
    raise AssertionError(f"case {case} index 0: expected {typ}, got {got!r}")


def naive(runs, dtype=None):
    if dtype is None:
        dtype = np.float64
    entries = []
    for r, run in enumerate(runs):
        for i, v in enumerate(run):
            entries.append((v.item(), r, i))
    entries.sort(key=lambda item: (item[0], item[1], item[2]))
    return np.asarray([v for v, _, _ in entries], dtype=dtype), np.asarray(
        [r for _, r, _ in entries], dtype=np.int64
    )


def verify(case, runs):
    snapshots = [x.copy() for x in runs]
    dtype = runs[0].dtype if runs else np.dtype(np.float64)
    exp_v, exp_o = naive(runs, dtype)
    got_v, got_o = merge_sorted_runs(iter(runs))
    check(case + "-values-shape", 0, exp_v.shape, got_v.shape)
    check(case + "-values-dtype", 0, dtype, got_v.dtype)
    check(case + "-origins-shape", 0, exp_o.shape, got_o.shape)
    check(case + "-origins-dtype", 0, np.dtype(np.int64), got_o.dtype)
    if not np.array_equal(exp_v, got_v):
        i = int(np.flatnonzero(exp_v != got_v)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp_v[i]!r}, got {got_v[i]!r}"
        )
    if not np.array_equal(exp_o, got_o):
        i = int(np.flatnonzero(exp_o != got_o)[0])
        raise AssertionError(
            f"case {case} origins index {i}: expected {exp_o[i]!r}, got {got_o[i]!r}"
        )
    check(
        case + "-independent-output",
        0,
        False,
        any(np.shares_memory(got_v, x) or np.shares_memory(got_o, x) for x in runs),
    )
    check(
        case + "-inputs-unchanged",
        0,
        True,
        all(np.array_equal(a, b) for a, b in zip(runs, snapshots)),
    )


verify("empty-iterable-contract", [])
verify(
    "stable-run-ties-and-empty-run",
    [
        np.array([5, 6, 6], np.int16),
        np.array([1, 6, 7], np.int16),
        np.array([], np.int16),
    ],
)
for bad in (
    [np.ones((2, 2), np.int32)],
    [np.array([1], np.int32), np.array([1], np.int64)],
    [np.array([True])],
    [np.array([1 + 2j])],
    [np.array(["a"])],
):
    error(
        "rank-or-common-real-dtype", TypeError, lambda bad=bad: merge_sorted_runs(bad)
    )
error(
    "unsorted-run", ValueError, lambda: merge_sorted_runs([np.array([2, 1], np.int32)])
)
# Fixed-seed independent stable key sort, 256 randomized run collections with duplicates and empty runs.
for case in range(256):
    nr = int(rng.integers(0, 8))
    dtype = (np.int16, np.float32, np.float64)[case % 3]
    runs = []
    for r in range(nr):
        n = int(rng.integers(0, 80))
        raw = (
            rng.integers(-20, 21, size=n).astype(dtype)
            if np.dtype(dtype).kind == "i"
            else rng.integers(-20, 21, size=n).astype(dtype)
        )
        runs.append(np.sort(raw))
    verify("random-stable-merge", runs)
# Largest practical aggregate input, tie-heavy and spread across 8 source runs.
runs = [np.sort(rng.integers(-500, 501, size=16384, dtype=np.int32)) for _ in range(8)]
verify("largest-merge", runs)
