import sys, struct
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


def naive(nums):
    out = bytearray(struct.pack("<h", int(nums[0])))
    for i in range(1, len(nums)):
        out.append((int(nums[i]) - int(nums[i - 1])) & 255)
    return bytes(out)


path = Path(__file__).resolve().parents[1] / "candidate.py"
template = Path(__file__).resolve().parent / "template.py"
check(
    "skeleton-fixed-lines",
    0,
    _without_todo(template.read_text()),
    _without_todo(path.read_text()),
)
from candidate import encode_delta_bytes


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "values",
    ", ".join(inspect.signature(encode_delta_bytes).parameters),
)
check(
    "little-endian-delta",
    0,
    bytes([254, 255, 3, 255]),
    encode_delta_bytes(np.array([-2, 1, 0], np.int16)),
)
check("single-value", 0, b"\x00\x80", encode_delta_bytes(np.array([-32768], np.int16)))
check(
    "positive-int16-endpoint",
    0,
    struct.pack("<h", 32767),
    encode_delta_bytes(np.array([32767], np.int64)),
)
for bad in (
    np.array([], np.int64),
    np.array(1, np.int64),
    np.array([[1, 2]], np.int64),
    np.array([1.0]),
    np.array([True]),
    np.array(["1"]),
):
    error(
        "invalid-rank-empty-or-dtype",
        ValueError,
        lambda bad=bad: encode_delta_bytes(bad),
    )
for bad in (np.array([32768], np.int64), np.array([-32769], np.int64)):
    error(
        "first-value-int16-range", ValueError, lambda bad=bad: encode_delta_bytes(bad)
    )
for bad in (np.array([0, 128], np.int64), np.array([0, -129], np.int64)):
    error("delta-int8-range", ValueError, lambda bad=bad: encode_delta_bytes(bad))
# Fixed seed 256 valid sequences. Python oracle computes each signed delta and packs low bytes independently.
rng = np.random.default_rng(20261105)
for case in range(256):
    n = int(rng.integers(1, 257))
    nums = [int(rng.integers(-32768, 32768))]
    for _ in range(n - 1):
        nums.append(nums[-1] + int(rng.integers(-128, 128)))
    arr = np.asarray(nums, dtype=np.int64)
    check("random-delta-encoding", case, naive(nums), encode_delta_bytes(arr))
# Largest practical sequence; byte count is linear and the first value exercises low signed encoding.
nums = [-32768]
for i in range(1, 131072):
    nums.append(nums[-1] + ((i % 256) - 128))
nums = np.asarray(nums, dtype=np.int64)
check("largest-delta-sequence", 0, naive(nums.tolist()), encode_delta_bytes(nums))
