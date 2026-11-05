import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import take_axis_positions


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "array, positions, axis",
    ", ".join(inspect.signature(take_axis_positions).parameters),
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


def naive_take(a, p, axis):
    shape = list(a.shape)
    shape[axis] = len(p)
    out = np.empty(shape, dtype=a.dtype)
    for oi in np.ndindex(tuple(shape)):
        ii = list(oi)
        ii[axis] = int(p[oi[axis]])
        out[oi] = a[tuple(ii)]
    return out


x = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
pos = np.array([2, 0, 2], dtype=np.uint8)
y = take_axis_positions(x, pos, -2)
check("gather-shape", 0, (2, 3, 4), y.shape)
check("gather-dtype", 0, x.dtype, y.dtype)
check(
    "gather-values-order-duplicates", 0, True, np.array_equal(y, naive_take(x, pos, 1))
)
check("gather-copy", 0, False, np.shares_memory(x, y))
y[0, 0, 0] = -1
check("gather-no-input-mutation", 0, 8, x[0, 2, 0])
empty = take_axis_positions(x, np.array([], dtype=np.int64), 1)
check("empty-positions-shape", 0, (2, 0, 4), empty.shape)
check("empty-positions-dtype", 0, x.dtype, empty.dtype)
error(
    "scalar-array",
    ValueError,
    lambda: take_axis_positions(np.array(1), np.array([0]), 0),
)
for axis in (True, 1.0, np.bool_(False)):
    error("axis-type", TypeError, lambda axis=axis: take_axis_positions(x, pos, axis))
for axis in (3, -4):
    error("axis-range", ValueError, lambda axis=axis: take_axis_positions(x, pos, axis))
for bad in (
    np.array([1.0]),
    np.array([True]),
    np.array([[1]], dtype=np.int64),
    np.array([], dtype=np.float64),
):
    error(
        "positions-rank-or-dtype",
        TypeError,
        lambda bad=bad: take_axis_positions(x, bad, 1),
    )
for bad in (np.array([-1]), np.array([3]), np.array([0, 99], dtype=np.uint64)):
    error(
        "position-bounds-no-negative-wrap",
        IndexError,
        lambda bad=bad: take_axis_positions(x, bad, 1),
    )
# Independent coordinate oracle over 256 fixed-seed shapes/axes and signed/unsigned empty/duplicate gathers.
for case in range(256):
    ndim = int(rng.integers(1, 5))
    shape = tuple(int(rng.integers(1, 8)) for _ in range(ndim))
    a = rng.integers(-100, 100, size=shape, dtype=np.int32)
    axis = int(rng.integers(-ndim, ndim))
    ax = axis % ndim
    npos = int(rng.integers(0, shape[ax] + 1))
    p = (
        rng.integers(0, shape[ax], size=npos, dtype=np.int64)
        if npos
        else np.array([], dtype=np.int64)
    )
    got = take_axis_positions(a, p, axis)
    exp = naive_take(a, p, ax)
    check("random-gather-shape", case, exp.shape, got.shape)
    check("random-gather-dtype", case, a.dtype, got.dtype)
    check("random-gather-values", case, True, np.array_equal(exp, got))
    check("random-gather-copy", case, False, np.shares_memory(a, got))
a = rng.integers(0, 100, size=(256, 32, 8), dtype=np.uint16)
p = np.arange(32, dtype=np.int32)[::-1]
got = take_axis_positions(a, p, 1)
check("largest-gather-values", 0, True, np.array_equal(naive_take(a, p, 1), got))
