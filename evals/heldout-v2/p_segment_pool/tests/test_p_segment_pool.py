import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import segment_reduce_sum


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "values, segment_ids, n_segments",
    ", ".join(inspect.signature(segment_reduce_sum).parameters),
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


def naive(v, ids, n):
    out = np.zeros((n,) + v.shape[1:], dtype=np.float64)
    for i, seg in enumerate(ids):
        out[int(seg)] += np.asarray(v[i], dtype=np.float64)
    return out


def verify(case, v, ids, n):
    vc = v.copy()
    ic = ids.copy()
    got = segment_reduce_sum(v, ids, n)
    exp = naive(v, ids, n)
    check(case + "-shape", 0, exp.shape, got.shape)
    check(case + "-dtype", 0, np.dtype(np.float64), got.dtype)
    if not np.array_equal(exp, got):
        bad = np.argwhere(exp != got)
        i = tuple(int(x) for x in bad[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp[i]!r}, got {got[i]!r}"
        )
    check(
        case + "-inputs-unchanged",
        0,
        True,
        np.array_equal(v, vc) and np.array_equal(ids, ic),
    )


v = np.array([[1, 2], [3, 4], [5, 6]], np.int16)
ids = np.array([2, 0, 2], np.uint8)
verify("leading-rows-and-empty-segments", v, ids, 4)
empty = np.empty((0, 2), np.float32)
verify("empty-input-valid", empty, np.array([], np.int64), 3)
verify("zero-segments-empty", np.empty((0,), np.float64), np.array([], np.int64), 0)
error(
    "empty-values-require-empty-ids",
    ValueError,
    lambda: segment_reduce_sum(np.empty((0, 2)), np.array([0], dtype=np.int64), 1),
)
error(
    "scalar-values",
    ValueError,
    lambda: segment_reduce_sum(np.array(1), np.array([], np.int64), 0),
)
error("id-rank", ValueError, lambda: segment_reduce_sum(v, np.array([[0, 1, 2]]), 3))
error("id-count", ValueError, lambda: segment_reduce_sum(v, np.array([0, 1]), 3))
for bad in (np.array([True]), np.array([1.0]), np.array([1 + 2j])):
    error(
        "id-integer-dtype",
        TypeError,
        lambda bad=bad: segment_reduce_sum(np.ones((1, 2)), bad, 2),
    )
for bad in (
    np.array([True]),
    np.array([1 + 2j]),
    np.array([object()], dtype=object),
    np.array(["x"]),
):
    error(
        "values-real-numeric-dtype",
        TypeError,
        lambda bad=bad: segment_reduce_sum(bad, np.array([0]), 1),
    )
for bad in (True, 1.5):
    error(
        "segment-count-type",
        TypeError,
        lambda bad=bad: segment_reduce_sum(empty, np.array([], np.int64), bad),
    )
for bad in (-1,):
    error(
        "segment-count-negative",
        ValueError,
        lambda bad=bad: segment_reduce_sum(empty, np.array([], np.int64), bad),
    )
error(
    "negative-segment-id",
    ValueError,
    lambda: segment_reduce_sum(v, np.array([0, -1, 1]), 3),
)
error(
    "segment-id-at-upper-bound",
    ValueError,
    lambda: segment_reduce_sum(v, np.array([0, 1, 3]), 3),
)
error(
    "nonempty-values-zero-segments",
    ValueError,
    lambda: segment_reduce_sum(np.ones((1,)), np.array([0]), 0),
)
# Fixed-seed naive leading-row accumulation over 256 sizes, ranks, and empty segments.
for case in range(256):
    rows = int(rng.integers(0, 129))
    nseg = int(rng.integers(0, 12))
    tail = () if case % 2 else (int(rng.integers(1, 5)), int(rng.integers(1, 4)))
    v = (
        rng.integers(-20, 21, size=(rows,) + tail, dtype=np.int16)
        if case % 3 == 0
        else rng.normal(size=(rows,) + tail).astype(np.float32)
    )
    ids = (
        rng.integers(0, nseg, size=rows, dtype=np.int64)
        if rows and nseg
        else np.array([], dtype=np.int64)
    )
    if rows and nseg == 0:
        rows = 0
        v = v[:0]
    verify("random-segment-sum", v, ids, nseg)
# Largest practical 65536-row matrix with eight features and many empty-capable segments.
v = rng.normal(size=(65536, 8)).astype(np.float32)
ids = rng.integers(0, 257, size=65536, dtype=np.int32)
verify("largest-segment-pool", v, ids, 257)
