import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import weighted_blend32


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "a, b, weight",
    ", ".join(inspect.signature(weighted_blend32).parameters),
)
rng = np.random.default_rng(20261105)
RTOL, ATOL = 1e-5, 1e-6


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


def close(case, exp, got):
    check(case + "-shape", 0, exp.shape, got.shape)
    check(case + "-dtype", 0, np.dtype(np.float32), got.dtype)
    bad = ~np.isclose(exp, got, rtol=RTOL, atol=ATOL)
    if np.any(bad):
        i = tuple(int(x) for x in np.argwhere(bad)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp[i]!r}, got {got[i]!r} (rtol={RTOL}, atol={ATOL})"
        )


def naive(a, b, w):
    aa, bb, ww = np.broadcast_arrays(np.asarray(a), np.asarray(b), np.asarray(w))
    out = np.empty(aa.shape, np.float32)
    for i in np.ndindex(aa.shape):
        out[i] = float(aa[i]) * float(ww[i]) + float(bb[i]) * (1.0 - float(ww[i]))
    return out


# Broadcasting, explicit float32 output, scalar/array weights, endpoint weights, and input immutability.
a = np.array([0, 10], np.int16)
b = np.array([[10], [20]], np.float64)
w = np.array([0.25, 0.5], np.float64)
ac = a.copy()
bc = b.copy()
wc = w.copy()
y = weighted_blend32(a, b, w)
close("broadcasting-float32", np.array([[7.5, 10.0], [15.0, 15.0]], np.float32), y)
check("input-a-unchanged", 0, True, np.array_equal(a, ac))
check("input-b-unchanged", 0, True, np.array_equal(b, bc))
check("weight-unchanged", 0, True, np.array_equal(w, wc))
close("weight-zero", np.array([2.0], np.float32), weighted_blend32([1], [2], 0))
close("weight-one", np.array([1.0], np.float32), weighted_blend32([1], [2], 1))
for name, bad in (
    ("bool-a", np.array([True])),
    ("complex-a", np.array([1 + 2j])),
    ("object-a", np.array([object()], dtype=object)),
    ("string-a", np.array(["1"])),
    ("bool-b", np.array([True])),
    ("complex-b", np.array([1 + 2j])),
    ("object-b", np.array([object()], dtype=object)),
    ("string-b", np.array(["1"])),
    ("bool-weight", True),
    ("complex-weight", 1 + 0j),
    ("string-weight", ".5"),
):
    error(
        name,
        TypeError,
        lambda bad=bad: weighted_blend32(bad, [1], 0.5)
        if name.endswith("a") or name.endswith("b")
        else weighted_blend32([1], [2], bad),
    )
for wbad in (-0.1, 1.1, np.nan, np.inf, -np.inf, np.array([0.2, 1.2])):
    error(
        "invalid-weight-range-or-finiteness",
        ValueError,
        lambda wbad=wbad: weighted_blend32([1, 2], [3, 4], wbad),
    )
error(
    "broadcast-shape-mismatch",
    ValueError,
    lambda: weighted_blend32(np.ones((2,)), np.ones((3,)), 0.5),
)
error(
    "weight-broadcast-mismatch",
    ValueError,
    lambda: weighted_blend32(np.ones((2, 3)), np.ones((2, 3)), np.ones((4,))),
)
# Fixed-seed broadcastable randomized oracle (256 cases), then a 524288-element case.
for case in range(256):
    base = (int(rng.integers(1, 6)), int(rng.integers(1, 6)))
    a_shape = (base[0], 1)
    b_shape = (1, base[1])
    aa = (
        rng.integers(-100, 101, size=a_shape, dtype=np.int16)
        if case % 2
        else rng.normal(size=a_shape).astype(np.float64)
    )
    bb = rng.normal(size=b_shape).astype(np.float32)
    ww = rng.random(size=base)
    got = weighted_blend32(aa, bb, ww)
    close("random-broadcast-blend", naive(aa, bb, ww), got)
aa = rng.normal(size=(1024, 512)).astype(np.float64)
bb = rng.normal(size=(1, 512)).astype(np.int16)
aa_before = aa.copy()
bb_before = bb.copy()
got = weighted_blend32(aa, bb, 0.375)
close("largest-broadcast", naive(aa, bb, 0.375), got)
check(
    "large-inputs-unchanged",
    0,
    True,
    np.array_equal(aa, aa_before) and np.array_equal(bb, bb_before),
)
