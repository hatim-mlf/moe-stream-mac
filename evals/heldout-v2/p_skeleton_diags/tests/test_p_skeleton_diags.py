import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect


def _without_todo(text):
    out = []
    inside = False
    for line in text.splitlines(keepends=True):
        if "TODO_BEGIN" in line:
            out.append(line)
            inside = True
        elif "TODO_END" in line:
            out.append(line)
            inside = False
        elif not inside:
            out.append(line)
    if inside:
        raise AssertionError(
            "case skeleton-fixed-lines index 0: expected closed TODO region, got unclosed"
        )
    return out


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


def naive(a):
    vals = []
    for off in range(-(a.shape[0] - 1), a.shape[1]):
        diag = []
        for r in range(a.shape[0]):
            c = r + off
            if 0 <= c < a.shape[1]:
                diag.append(float(a[r, c]))
        vals.append(sum(diag) / len(diag))
    return np.asarray(vals, dtype=np.float64)


def verify(case, a):
    got = diagonal_means(a)
    exp = naive(a)
    check(case + "-rank", 0, 1, got.ndim)
    check(case + "-dtype", 0, np.dtype(np.float64), got.dtype)
    check(case + "-shape", 0, exp.shape, got.shape)
    bad = ~np.isclose(exp, got, rtol=1e-5, atol=1e-6)
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp[i]!r}, got {got[i]!r} (rtol=1e-5, atol=1e-6)"
        )


path = Path(__file__).resolve().parents[1] / "candidate.py"
template = Path(__file__).resolve().parent / "template.py"
check(
    "skeleton-fixed-lines",
    0,
    _without_todo(template.read_text()),
    _without_todo(path.read_text()),
)
from candidate import diagonal_means


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "matrix",
    ", ".join(inspect.signature(diagonal_means).parameters),
)
verify("rectangular-diagonal-order", np.array([[1, 2, 3], [4, 5, 6]], np.int16))
verify("single-row", np.array([[1.0, 2.0, 4.0]]))
verify("single-column", np.array([[1.0], [3.0], [5.0]]))
for bad in (np.empty((0, 2)), np.empty((2, 0)), np.array([1, 2]), np.array(1)):
    error("rank-or-empty-shape", ValueError, lambda bad=bad: diagonal_means(bad))
for bad in (
    np.array([[True]]),
    np.array([[1 + 2j]]),
    np.array([[object()]], object),
    np.array([["1"]]),
    np.array([[np.nan]]),
    np.array([[np.inf]]),
    np.array([[-np.inf]]),
):
    error("non-real-or-nonfinite", TypeError, lambda bad=bad: diagonal_means(bad))
# Fixed-seed independent coordinate enumeration for 256 shapes, plus a 512x512 stress matrix.
rng = np.random.default_rng(20261105)
for case in range(256):
    rows = int(rng.integers(1, 18))
    cols = int(rng.integers(1, 20))
    a = (
        rng.integers(-100, 101, size=(rows, cols), dtype=np.int16)
        if case % 2
        else rng.normal(size=(rows, cols)).astype(np.float32)
    )
    verify("random-diagonal-means", a)
a = rng.normal(size=(512, 512)).astype(np.float64)
verify("largest-diagonal-matrix", a)
