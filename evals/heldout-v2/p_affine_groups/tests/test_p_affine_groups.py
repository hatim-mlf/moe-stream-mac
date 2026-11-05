import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import dequant_byte_groups


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "packed, scales, biases, group_size",
    ", ".join(inspect.signature(dequant_byte_groups).parameters),
)
RTOL, ATOL = 1e-5, 1e-6
rng = np.random.default_rng(20261105)


def check(case, index, expected, got):
    if expected != got:
        raise AssertionError(
            f"case {case} index {index}: expected {expected!r}, got {got!r}"
        )


def array_close(case, expected, got):
    check(case + ".shape", 0, expected.shape, got.shape)
    check(case + ".dtype", 0, np.dtype(np.float32), got.dtype)
    ok = np.isclose(expected, got, rtol=RTOL, atol=ATOL, equal_nan=True)
    if not np.all(ok):
        i = tuple(int(x) for x in np.argwhere(~ok)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {expected[i]!r}, got {got[i]!r} (rtol={RTOL}, atol={ATOL})"
        )


def error(case, typ, fn):
    try:
        got = fn()
    except Exception as exc:
        check(case, 0, typ, type(exc))
        return
    raise AssertionError(f"case {case} index 0: expected {typ}, got return {got!r}")


def naive(p, scales, biases, group):
    q = []
    for byte in p:
        q.extend([int(byte) & 15, int(byte) >> 4])
    return np.asarray(
        [
            q[i] * float(scales[i // group]) + float(biases[i // group])
            for i in range(len(q))
        ],
        dtype=np.float32,
    )


# Named contract cases: unsigned nibbles, every group has its own scale and bias.
p = np.array([0xF1, 0xA3], dtype=np.uint8)
s = np.array([2, -1])
b = np.array([0.5, 1])
y = dequant_byte_groups(p, s, b, 2)
array_close(
    "unsigned-high-nibbles-distinct-groups",
    np.array([2.5, 30.5, -2, -9], np.float32),
    y,
)
array_close(
    "numpy-integer-group-size",
    np.array([1, 15, 3, 10], np.float32),
    dequant_byte_groups(p, np.ones(2), np.zeros(2), np.int64(2)),
)
check(
    "empty-packed-vector",
    0,
    (0,),
    dequant_byte_groups(
        np.array([], np.uint8), np.array([], np.float32), np.array([], np.float32), 3
    ).shape,
)
array_close(
    "empty-packed-vector-values",
    np.array([], np.float32),
    dequant_byte_groups(
        np.array([], np.uint8), np.array([], np.float32), np.array([], np.float32), 3
    ),
)
# Type vs layout errors, including bool-vs-int group sizes.
error(
    "packed-rank",
    TypeError,
    lambda: dequant_byte_groups(
        np.zeros((1, 1), np.uint8), np.array([1]), np.array([0]), 2
    ),
)
error(
    "packed-dtype",
    TypeError,
    lambda: dequant_byte_groups(
        np.array([1], np.int8), np.array([1]), np.array([0]), 2
    ),
)
error(
    "scale-rank",
    ValueError,
    lambda: dequant_byte_groups(
        np.array([1], np.uint8), np.array([[1]]), np.array([0]), 2
    ),
)
error(
    "bias-rank",
    ValueError,
    lambda: dequant_byte_groups(
        np.array([1], np.uint8), np.array([1]), np.array([[0]]), 2
    ),
)
error(
    "scale-bias-shape",
    ValueError,
    lambda: dequant_byte_groups(
        np.array([1], np.uint8), np.array([1, 2]), np.array([0]), 2
    ),
)
for name, bad in (
    ("scale-bool-dtype", np.array([True])),
    ("scale-complex-dtype", np.array([1 + 2j])),
    ("bias-object-dtype", np.array([object()], dtype=object)),
):
    error(
        name,
        TypeError,
        lambda bad=bad: dequant_byte_groups(
            np.array([1], np.uint8), bad, np.array([0]), 2
        ),
    )
for bad in (True, 2.0, "2"):
    error(
        "group-size-type", TypeError, lambda bad=bad: dequant_byte_groups(p, s, b, bad)
    )
for bad in (0, -1):
    error(
        "group-size-range",
        ValueError,
        lambda bad=bad: dequant_byte_groups(p, s, b, bad),
    )
error(
    "unpacked-not-divisible",
    ValueError,
    lambda: dequant_byte_groups(
        np.array([1], np.uint8), np.array([1]), np.array([0]), 3
    ),
)
error(
    "group-count-mismatch",
    ValueError,
    lambda: dequant_byte_groups(p, np.array([1]), np.array([0]), 2),
)
# Fixed-seed independent scalar oracle: 256 cases, odd lengths, multiple groups and a 131072-value stress case.
for case in range(256):
    group = int(rng.integers(1, 18))
    groups = int(rng.integers(1, 12))
    if group % 2 and groups % 2:
        groups += 1
    n = group * groups
    packed = rng.integers(0, 256, size=(n + 1) // 2, dtype=np.uint8)
    scales = rng.normal(size=groups).astype(np.float64)
    biases = rng.normal(size=groups).astype(np.float32)
    got = dequant_byte_groups(packed, scales, biases, group)
    array_close("random-groups", naive(packed, scales, biases, group), got)
# Largest practical byte vector for this pure NumPy routine; 65536 bytes -> 131072 unpacked values.
packed = rng.integers(0, 256, size=65536, dtype=np.uint8)
scales = rng.normal(size=256)
biases = rng.normal(size=256)
group = 512
array_close(
    "largest-vector",
    naive(packed, scales, biases, group),
    dequant_byte_groups(packed, scales, biases, group),
)
