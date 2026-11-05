import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import window_matrix_copy


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "values, width, step",
    ", ".join(inspect.signature(window_matrix_copy).parameters),
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


def naive(a, w, step):
    starts = range(0, a.shape[0] - w + 1, step)
    out = np.empty((len(starts), w) + a.shape[1:], dtype=a.dtype)
    for wi, start in enumerate(starts):
        for j in range(w):
            out[(wi, j)] = a[(start + j,)]
    return out


def verify(case, a, w, step):
    before = a.copy()
    got = window_matrix_copy(a, w, step)
    exp = naive(a, w, step)
    check(case + "-shape", 0, exp.shape, got.shape)
    check(case + "-dtype", 0, a.dtype, got.dtype)
    check(case + "-copy", 0, False, np.shares_memory(a, got))
    if not np.array_equal(exp, got):
        i = tuple(int(x) for x in np.argwhere(exp != got)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp[i]!r}, got {got[i]!r}"
        )
    check(case + "-input-unchanged", 0, True, np.array_equal(a, before))


x = np.arange(20, dtype=np.int16).reshape(5, 2, 2)
verify("leading-axis-step", x, 3, 2)
verify("single-window-width-n", x, 5, 1)
verify("step-larger-than-window", x, 1, 4)
y = window_matrix_copy(x, 3, 2)
y[0, 0, 0, 0] = -9
check("mutating-result-does-not-mutate-input", 0, 0, x[0, 0, 0])
error("scalar-rank", ValueError, lambda: window_matrix_copy(np.array(1), 1, 1))
for w, s in ((True, 1), (2, False), (1.0, 1), (1, np.float64(1))):
    error(
        "width-step-type-bool-vs-int",
        TypeError,
        lambda w=w, s=s: window_matrix_copy(x, w, s),
    )
for w, s in ((0, 1), (-1, 1), (6, 1), (1, 0), (1, -1)):
    error("width-step-range", ValueError, lambda w=w, s=s: window_matrix_copy(x, w, s))
# Fixed-seed independent element copy over 256 shapes, dtypes, widths, and steps.
for case in range(256):
    leading = int(rng.integers(1, 20))
    tail = (int(rng.integers(1, 5)),) if case % 2 else ()
    dtype = (np.int16, np.float32, np.uint8)[case % 3]
    a = rng.integers(0, 100, size=(leading,) + tail).astype(dtype)
    w = int(rng.integers(1, leading + 1))
    step = int(rng.integers(1, leading + 4))
    verify("random-window-copy", a, w, step)
# Largest practical leading-axis window set (overlapping windows produce >1M copied values).
a = rng.integers(0, 256, size=(8192, 16), dtype=np.uint8)
verify("largest-window-copy", a, 128, 1)
