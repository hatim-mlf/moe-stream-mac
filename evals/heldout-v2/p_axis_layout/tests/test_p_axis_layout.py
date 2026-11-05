import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import move_feature_axis


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "values, axis, destination",
    ", ".join(inspect.signature(move_feature_axis).parameters),
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


def naive_move(a, src, dst):
    order = list(range(a.ndim))
    axis = order.pop(src)
    order.insert(dst, axis)
    shape = tuple(a.shape[k] for k in order)
    out = np.empty(shape, dtype=a.dtype)
    for oi in np.ndindex(shape):
        ii = [0] * a.ndim
        for od, old_axis in enumerate(order):
            ii[old_axis] = oi[od]
        out[oi] = a[tuple(ii)]
    return out


# Conversion, rank validation, positive/negative normalization, and moveaxis semantics.
x = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
y = move_feature_axis(x, 0, -1)
check("axis-move-shape", 0, (3, 4, 2), y.shape)
check("axis-move-dtype", 0, x.dtype, y.dtype)
check("axis-move-contiguous", 0, True, y.flags.c_contiguous)
check("axis-move-values", 0, True, np.array_equal(y, naive_move(x, 0, 2)))
check("axis-move-independent", 0, False, np.shares_memory(x, y))
y[0, 0, 0] = -99
check("axis-move-no-input-mutation", 0, 0, x[0, 0, 0])
check(
    "array-like-conversion",
    0,
    (3, 2),
    move_feature_axis([[1, 2, 3], [4, 5, 6]], 0, 1).shape,
)
error("scalar-rank", ValueError, lambda: move_feature_axis(np.array(4), 0, 0))
empty = np.empty((2, 0, 3), dtype=np.float32)
moved = move_feature_axis(empty, 2, 0)
check("empty-axis-shape", 0, (3, 2, 0), moved.shape)
check("empty-axis-dtype", 0, empty.dtype, moved.dtype)
check("empty-axis-contiguous", 0, True, moved.flags.c_contiguous)
for axis, dest in ((True, 0), (0, False), (1.0, 0), (0, np.float64(1))):
    error(
        "axis-type-bool-or-nonint",
        TypeError,
        lambda axis=axis, dest=dest: move_feature_axis(x, axis, dest),
    )
for axis, dest in ((3, 0), (0, -4), (-4, 0), (0, 3)):
    error(
        "axis-range",
        ValueError,
        lambda axis=axis, dest=dest: move_feature_axis(x, axis, dest),
    )
# Fixed-seed independent coordinate-map oracle, including singleton/empty dimensions and practical large data.
for case in range(256):
    ndim = int(rng.integers(1, 5))
    shape = tuple(int(rng.integers(1, 5)) for _ in range(ndim))
    a = rng.integers(-50, 51, size=shape, dtype=np.int32)
    src = int(rng.integers(-ndim, ndim))
    dst = int(rng.integers(-ndim, ndim))
    ns = src % ndim
    nd = dst % ndim
    got = move_feature_axis(a, src, dst)
    exp = naive_move(a, ns, nd)
    check("random-axis-move-shape", case, exp.shape, got.shape)
    check("random-axis-move-dtype", case, a.dtype, got.dtype)
    check("random-axis-move-values", case, True, np.array_equal(exp, got))
    check("random-axis-move-copy", case, False, np.shares_memory(a, got))
    check("random-axis-move-contiguous", case, True, got.flags.c_contiguous)
a = rng.normal(size=(64, 32, 16)).astype(np.float32)
got = move_feature_axis(a, 2, 0)
check("largest-axis-move-values", 0, True, np.array_equal(naive_move(a, 2, 0), got))
check("largest-axis-move-contiguous", 0, True, got.flags.c_contiguous)
