import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import masked_scale_copy


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "values, mask, factor",
    ", ".join(inspect.signature(masked_scale_copy).parameters),
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


def naive(a, m, f):
    mb = np.broadcast_to(m, a.shape)
    out = np.empty_like(a)
    for i in np.ndindex(a.shape):
        out[i] = a[i] * f if mb[i] else a[i]
    return out


def verify(case, a, m, f):
    ac = a.copy()
    mc = m.copy()
    got = masked_scale_copy(a, m, f)
    exp = naive(a, m, f)
    check(case + "-shape", 0, a.shape, got.shape)
    check(case + "-dtype", 0, a.dtype, got.dtype)
    check(case + "-copy", 0, False, np.shares_memory(a, got))
    bad = ~np.isclose(exp, got, rtol=RTOL, atol=ATOL, equal_nan=True)
    if np.any(bad):
        i = tuple(int(x) for x in np.argwhere(bad)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp[i]!r}, got {got[i]!r} "
            f"(rtol={RTOL}, atol={ATOL})"
        )
    check(
        case + "-inputs-unchanged",
        0,
        True,
        np.array_equal(a, ac) and np.array_equal(m, mc),
    )


x = np.array([[1, 2], [3, 4]], dtype=np.float32)
m = np.array([True, False])
y = masked_scale_copy(x, m, 2.5)
check("broadcast-mask-values", 0, True, np.array_equal(y, [[2.5, 2], [7.5, 4]]))
check("float32-copy", 0, np.dtype(np.float32), y.dtype)
check("independent-copy", 0, False, np.shares_memory(x, y))
y[0, 0] = 99
check("no-input-mutation", 0, 1.0, x[0, 0])
for bad in (
    np.array([1], np.int32),
    np.array([True]),
    np.array([1 + 2j]),
    np.array(["x"]),
):
    error(
        "values-floating-dtype",
        TypeError,
        lambda bad=bad: masked_scale_copy(bad, np.array([True]), 2),
    )
for bad in (np.array([1], dtype=np.int8), np.array([1.0]), np.array([1 + 0j])):
    error("mask-bool-dtype", TypeError, lambda bad=bad: masked_scale_copy(x, bad, 2))
for bad in ([2, 3], np.nan, np.inf, -np.inf, True, 1 + 2j, "2"):
    error(
        "factor-finite-real-scalar",
        ValueError,
        lambda bad=bad: masked_scale_copy(x, m, bad),
    )
error(
    "mask-broadcast",
    ValueError,
    lambda: masked_scale_copy(np.ones((2, 3), np.float32), np.ones((4,), bool), 2),
)
# Fixed-seed per-element oracle across all supported float widths, broadcasting masks and scalar factors.
for case in range(256):
    dtype = (np.float16, np.float32, np.float64)[case % 3]
    shape = (int(rng.integers(1, 8)), int(rng.integers(1, 6)))
    a = rng.normal(size=shape).astype(dtype)
    m = rng.integers(0, 2, size=(shape[1],), dtype=np.uint8).astype(bool)
    f = float(rng.normal())
    verify("random-masked-copy", a, m, f)
# Largest practical array for this elementwise operation.
a = rng.normal(size=(1024, 512)).astype(np.float32)
m = rng.random(size=(1, 512)) > 0.5
verify("largest-mask-array", a, m, -1.25)
