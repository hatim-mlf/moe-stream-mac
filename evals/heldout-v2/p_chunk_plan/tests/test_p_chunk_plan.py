import sys, random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import plan_byte_chunks


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-public-signature",
    "total_bytes, chunk_bytes, alignment",
    ", ".join(inspect.signature(plan_byte_chunks).parameters),
)
rng = random.Random(20261105)


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


def naive(total, chunk):
    return [(off, min(chunk, total - off)) for off in range(0, total, chunk)]


def validate(case, total, chunk, align, got):
    check(case + "-list-type", 0, list, type(got))
    check(case + "-exact-coverage", 0, naive(total, chunk), got)
    cursor = 0
    for i, (off, size) in enumerate(got):
        check(
            case + "-pair-type",
            i,
            True,
            type((off, size)) is tuple and type(off) is int and type(size) is int,
        )
        check(case + "-offset", i, cursor, off)
        check(case + "-aligned-offset", i, 0, off % align)
        check(case + "-size-bound", i, True, 0 < size <= chunk)
        if i < len(got) - 1:
            check(case + "-nonfinal-alignment", i, 0, size % align)
        cursor += size
    check(case + "-end", len(got), total, cursor)


validate("aligned-short-tail", 25, 8, 4, plan_byte_chunks(25, 8, 4))
validate("zero-total", 0, 8, 4, plan_byte_chunks(0, 8, 4))
for arg in (True, False, 1.0, np.int64(2), "2"):
    error(
        "python-int-type-only", TypeError, lambda arg=arg: plan_byte_chunks(arg, 8, 4)
    )
for args in ((-1, 8, 4), (1, 0, 4), (1, 8, 0), (10, 7, 4)):
    error("range-and-alignment", ValueError, lambda args=args: plan_byte_chunks(*args))
for case in range(256):
    align = rng.randint(1, 32)
    chunk = align * rng.randint(1, 32)
    total = rng.randint(0, 10000)
    validate(
        "random-chunk-plan", total, chunk, align, plan_byte_chunks(total, chunk, align)
    )
validate("largest-plan", 1_000_003, 4096, 16, plan_byte_chunks(1_000_003, 4096, 16))
