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


def close(case, exp, got):
    check(case + "-shape", 0, exp.shape, got.shape)
    check(case + "-dtype", 0, np.dtype(np.float32), got.dtype)
    bad = ~np.isclose(exp, got, rtol=1e-5, atol=1e-6)
    if np.any(bad):
        i = tuple(int(x) for x in np.argwhere(bad)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {exp[i]!r}, got {got[i]!r} (rtol=1e-5, atol=1e-6)"
        )


def naive(raw, t, o):
    a = np.asarray(raw)
    out = np.empty(a.shape, np.float32)
    for i in np.ndindex(a.shape):
        z = (float(a[i]) - float(o)) / float(t)
        if z >= 0:
            v = 1 / (1 + np.exp(-z))
        else:
            e = np.exp(z)
            v = e / (1 + e)
        out[i] = v
    return out


path = Path(__file__).resolve().parents[1] / "candidate.py"
template = Path(__file__).resolve().parent / "template.py"
check(
    "skeleton-fixed-lines",
    0,
    _without_todo(template.read_text()),
    _without_todo(path.read_text()),
)
from candidate import calibrate_scores


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "raw, temperature, offset",
    ", ".join(inspect.signature(calibrate_scores).parameters),
)
x = np.array([-1000.0, 0.0, 2.0, 1000.0], np.float64)
close("stable-sigmoid-positive-negative", naive(x, 2, 1), calibrate_scores(x, 2, 1))
close(
    "scalar-input-shape",
    np.asarray(0.5, np.float32),
    calibrate_scores(np.asarray(0.0), 1.0, 0.0),
)
for raw in ([np.nan], [np.inf], [-np.inf], [1 + 2j], ["x"], [True], [object()]):
    error("invalid-raw-values", ValueError, lambda raw=raw: calibrate_scores(raw, 1, 0))
for t, o in (
    (0, 0),
    (-1, 0),
    (np.nan, 0),
    (np.inf, 0),
    (1, np.nan),
    (1, np.inf),
    (1, -np.inf),
):
    error(
        "invalid-scalar-parameters",
        ValueError,
        lambda t=t, o=o: calibrate_scores([1], t, o),
    )
for t, o in (([1], 0), (1, [0]), (np.array([1]), 0), (1, np.array([0]))):
    error(
        "nonscalar-parameters", TypeError, lambda t=t, o=o: calibrate_scores([1], t, o)
    )
# Fixed seed: 256 arrays of several ranks and magnitudes; oracle uses stable scalar branches.
rng = np.random.default_rng(20261105)
for case in range(256):
    shape = (
        (0,)
        if case % 37 == 0
        else (2, 0)
        if case % 41 == 0
        else ()
        if case % 11 == 0
        else (int(rng.integers(1, 8)),)
        if case % 3
        else (int(rng.integers(1, 6)), int(rng.integers(1, 5)))
    )
    raw = rng.normal(size=shape).astype((np.int32, np.float32, np.float64)[case % 3])
    if case % 7 == 0:
        raw = np.asarray(raw) * 1000
    t = float(rng.uniform(0.01, 10))
    off = float(rng.normal())
    close(
        "random-logistic-calibration", naive(raw, t, off), calibrate_scores(raw, t, off)
    )
# Largest practical shape for stable vectorized calibration.
raw = np.linspace(-1000, 1000, 262147, dtype=np.float64)
close("largest-score-array", naive(raw, 1.75, -3.5), calibrate_scores(raw, 1.75, -3.5))
