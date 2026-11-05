import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import stable_rank_rows


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "scores, group_ids, limit",
    ", ".join(inspect.signature(stable_rank_rows).parameters),
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


def naive(s, g, k):
    return np.asarray(
        sorted(range(len(s)), key=lambda i: (-float(s[i]), int(g[i]), i))[:k],
        dtype=np.int64,
    )


def verify(case, s, g, k):
    before_s = s.copy()
    before_g = g.copy()
    got = stable_rank_rows(s, g, k)
    exp = naive(s, g, k)
    check(case + "-shape", 0, (k,), got.shape)
    check(case + "-dtype", 0, np.dtype(np.int64), got.dtype)
    if not np.array_equal(exp, got):
        i = next(i for i, (x, y) in enumerate(zip(exp, got)) if x != y)
        raise AssertionError(
            f"case {case} index {i}: expected {int(exp[i])}, got {int(got[i])}"
        )
    check(case + "-scores-unchanged", 0, True, np.array_equal(s, before_s))
    check(case + "-groups-unchanged", 0, True, np.array_equal(g, before_g))


s = np.array([4, 8, 8, 8, 1], np.float32)
g = np.array([0, 2, 1, 1, 0], np.int64)
verify("score-group-source-ties", s, g, 5)
verify("zero-limit", s, g, 0)
verify("full-limit", s, g, 5)
error(
    "score-rank",
    ValueError,
    lambda: stable_rank_rows(np.ones((2, 2)), np.ones(4, dtype=np.int64), 2),
)
error(
    "group-rank",
    ValueError,
    lambda: stable_rank_rows(np.ones(2), np.ones((1, 2), dtype=np.int64), 2),
)
error(
    "length-mismatch",
    ValueError,
    lambda: stable_rank_rows(np.ones(3), np.ones(2, dtype=np.int64), 2),
)
for bad in (
    np.array([True]),
    np.array([1 + 2j]),
    np.array(["1"]),
    np.array([object()], dtype=object),
):
    error(
        "score-dtype",
        TypeError,
        lambda bad=bad: stable_rank_rows(bad, np.array([0]), 0),
    )
for bad in (np.array([True]), np.array([1.0]), np.array([1 + 2j])):
    error(
        "group-dtype",
        TypeError,
        lambda bad=bad: stable_rank_rows(np.array([1.0]), bad, 0),
    )
for bad in (True, 1.0, np.bool_(False)):
    error(
        "limit-type-bool-vs-int", TypeError, lambda bad=bad: stable_rank_rows(s, g, bad)
    )
for bad in (-1, 6):
    error("limit-range", ValueError, lambda bad=bad: stable_rank_rows(s, g, bad))
for bad in (np.array([np.nan]), np.array([np.inf]), np.array([-np.inf])):
    error(
        "nonfinite-score",
        ValueError,
        lambda bad=bad: stable_rank_rows(bad, np.array([0]), 1),
    )
# Fixed-seed independent Python stable-sort oracle, 256 cases with frequent ties and n=0 edge coverage.
for case in range(256):
    n = int(rng.integers(0, 257))
    scores = (
        rng.integers(-8, 9, size=n, dtype=np.int32)
        if case % 2
        else rng.normal(size=n).astype(np.float64)
    )
    groups = rng.integers(-5, 6, size=n, dtype=np.int16)
    k = int(rng.integers(0, n + 1))
    verify("random-stable-ranking", scores, groups, k)
# Largest practical row set, intentionally tie-heavy to stress all tie keys.
n = 65536
scores = rng.integers(-32, 33, size=n, dtype=np.int32)
groups = rng.integers(-100, 101, size=n, dtype=np.int64)
verify("largest-ranking", scores, groups, n)
