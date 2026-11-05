import sys, struct
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import unpack_frame_header


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "data",
    ", ".join(inspect.signature(unpack_frame_header).parameters),
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


def naive(blob):
    return (
        int.from_bytes(blob[5:6], "little"),
        int.from_bytes(blob[6:8], "little"),
        int.from_bytes(blob[8:12], "little"),
    )


blob = struct.pack("<4sBBHI", b"RBUF", 2, 5, 3, 0x12345678) + b"abc"
for name, kind in (
    ("bytes-input", bytes),
    ("bytearray-input", bytearray),
    ("memoryview-input", memoryview),
):
    value = kind(blob)
    before = bytes(value)
    got = unpack_frame_header(value)
    check(name + "-tuple", 0, (5, 3, 0x12345678), got)
    check(name + "-python-int-types", 0, True, all(type(v) is int for v in got))
    check(name + "-no-mutation", 0, before, bytes(value))
# Every malformed class has its own exception case: truncation, magic, version, flags, short/extra payload.
for n in range(12):
    error("truncated-header", ValueError, lambda n=n: unpack_frame_header(blob[:n]))
for name, bad in (
    ("bad-magic", struct.pack("<4sBBHI", b"NOPE", 2, 0, 0, 0)),
    ("bad-version", struct.pack("<4sBBHI", b"RBUF", 1, 0, 0, 0)),
    ("reserved-flags", struct.pack("<4sBBHI", b"RBUF", 2, 8, 0, 0)),
    ("payload-too-short", blob[:-1]),
    ("payload-extra", blob + b"x"),
):
    error(name, ValueError, lambda bad=bad: unpack_frame_header(bad))
for bad in ("text", [1, 2], np.array([1], dtype=np.uint8)):
    error("wrong-input-type", TypeError, lambda bad=bad: unpack_frame_header(bad))
zero = struct.pack("<4sBBHI", b"RBUF", 2, 7, 0, 0)
check("zero-payload", 0, (7, 0, 0), unpack_frame_header(zero))
# 256 fixed-seed frames, varying legal flags, lengths, and sequence values; parse fields independently with int.from_bytes.
for case in range(256):
    flags = int(rng.integers(0, 8))
    n = int(rng.integers(0, 1025))
    seq = int(rng.integers(0, 2**32, dtype=np.uint64))
    payload = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
    frame = struct.pack("<4sBBHI", b"RBUF", 2, flags, n, seq) + payload
    check("random-frame", case, naive(frame), unpack_frame_header(frame))
# Largest representable uint16 payload (65535 bytes), with all flag bits enabled.
n = 65535
frame = struct.pack("<4sBBHI", b"RBUF", 2, 7, n, 0xFFFFFFFF) + bytes(
    (i % 251 for i in range(n))
)
check("largest-payload", 0, (7, n, 0xFFFFFFFF), unpack_frame_header(memoryview(frame)))
