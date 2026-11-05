import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import inspect
from candidate import BudgetArrayCache


def signature_check(case, expected, got):
    if expected != got:
        raise AssertionError(f"case {case} index 0: expected {expected!r}, got {got!r}")


signature_check(
    "exact-constructor-signature",
    "self, budget_bytes",
    ", ".join(inspect.signature(BudgetArrayCache.__init__).parameters),
)
signature_check(
    "exact-entry-count-signature",
    "self",
    ", ".join(inspect.signature(BudgetArrayCache.entry_count).parameters),
)
signature_check(
    "exact-bytes-property-signature",
    "self",
    ", ".join(inspect.signature(BudgetArrayCache.bytes_used.fget).parameters),
)
signature_check(
    "exact-fetch-signature",
    "self, key",
    ", ".join(inspect.signature(BudgetArrayCache.fetch_cached).parameters),
)
signature_check(
    "exact-store-signature",
    "self, key, array",
    ", ".join(inspect.signature(BudgetArrayCache.store_cached).parameters),
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


def arr_eq(case, expected, got):
    check(case + "-dtype", 0, expected.dtype, got.dtype)
    check(case + "-shape", 0, expected.shape, got.shape)
    if not np.array_equal(expected, got):
        i = tuple(int(x) for x in np.argwhere(expected != got)[0])
        raise AssertionError(
            f"case {case} index {i}: expected {expected[i]!r}, got {got[i]!r}"
        )


class NaiveLRU:
    def __init__(self, budget):
        self.budget = budget
        self.items = []
        self.used = 0

    def fetch(self, key):
        for i, (k, a) in enumerate(self.items):
            if k == key:
                self.items.append(self.items.pop(i))
                return self.items[-1][1].copy()
        return None

    def store(self, key, a):
        for i, (k, old) in enumerate(self.items):
            if k == key:
                self.used -= old.nbytes
                self.items.pop(i)
                break
        value = np.array(a, copy=True, order="C")
        if value.nbytes > self.budget:
            return False
        while self.items and self.used + value.nbytes > self.budget:
            _, old = self.items.pop(0)
            self.used -= old.nbytes
        self.items.append((key, value))
        self.used += value.nbytes
        return True


for bad in (-1, True, False, 1.5, np.int64(4), "4"):
    error(
        "budget-nonnegative-python-int",
        ValueError,
        lambda bad=bad: BudgetArrayCache(bad),
    )
# zero budget, counts, exact byte accounting, replacement, oversized insertion, and LRU promotion/eviction.
c = BudgetArrayCache(20)
a = np.arange(3, dtype=np.int32)
store_a = c.store_cached("a", a)
check("store-a", 0, True, store_a)
check("store-return-exact-bool", 0, True, type(store_a) is bool)
check("bytes-after-a", 0, 12, c.bytes_used)
check("bytes-used-is-python-int", 0, True, type(c.bytes_used) is int)
check("count-after-a", 0, 1, c.entry_count())
check("entry-count-is-python-int", 0, True, type(c.entry_count()) is int)
check("store-b", 0, True, c.store_cached("b", np.array([8, 9], np.int32)))
check("bytes-after-b", 0, 20, c.bytes_used)
check("fetch-a-promotes", 0, True, np.array_equal(c.fetch_cached("a"), a))
check("store-c-evicts-lru-b", 0, True, c.store_cached("c", np.array([1], np.int32)))
check("bytes-after-eviction", 0, 16, c.bytes_used)
check("entry-count-after-eviction", 0, 2, c.entry_count())
check("b-evicted", 0, None, c.fetch_cached("b"))
check("a-kept", 0, True, np.array_equal(c.fetch_cached("a"), a))
copy = c.fetch_cached("a")
copy[0] = 999
check("fetch-returns-copy", 0, 0, c.fetch_cached("a")[0])
source = np.arange(6, dtype=np.int16).reshape(2, 3)[:, ::2]
check("store-strided-array", 0, True, c.store_cached("strided", source))
stored = c.fetch_cached("strided")
check("store-makes-contiguous-copy", 0, True, stored.flags.c_contiguous)
source[0, 0] = 88
check("store-isolated-from-source", 0, 0, stored[0, 0])
oldbytes = c.bytes_used
oldcount = c.entry_count()
check("oversized-not-cached", 0, False, c.store_cached("huge", np.zeros(100, np.uint8)))
check("oversized-absent", 0, None, c.fetch_cached("huge"))
check(
    "oversized-keeps-other-entries",
    0,
    (oldbytes, oldcount),
    (c.bytes_used, c.entry_count()),
)
# A replacement larger than the total budget first removes its previous value as specified.
z = BudgetArrayCache(8)
z.store_cached("key", np.ones(2, np.int32))
check(
    "oversized-replacement-result",
    0,
    False,
    z.store_cached("key", np.ones(3, np.int32)),
)
check("oversized-replacement-removes-old", 0, (0, 0), (z.bytes_used, z.entry_count()))
zero = BudgetArrayCache(0)
check(
    "zero-budget-empty-array",
    0,
    True,
    zero.store_cached("empty", np.empty((0,), np.float64)),
)
check("zero-budget-stats", 0, (0, 1), (zero.bytes_used, zero.entry_count()))
error("unhashable-key", TypeError, lambda: c.store_cached([], np.array([1])))
for bad in ([1, 2], memoryview(b"a")):
    error("ndarray-required", TypeError, lambda bad=bad: c.store_cached("bad", bad))
# 256 deterministic randomized operations checked against a separate list-based LRU model.
for case in range(256):
    budget = int(rng.integers(0, 4097))
    actual = BudgetArrayCache(budget)
    model = NaiveLRU(budget)
    for op in range(64):
        key = int(rng.integers(0, 12))
        if rng.random() < 0.58:
            shape = (int(rng.integers(0, 9)), int(rng.integers(1, 5)))
            dtype = (np.uint8, np.int16, np.float32)[int(rng.integers(0, 3))]
            data = rng.integers(0, 20, size=shape).astype(dtype)
            expected = model.store(key, data)
            got = actual.store_cached(key, data)
            check("random-store-result", (case, op), expected, got)
            check("random-store-bool-type", (case, op), True, type(got) is bool)
        else:
            expected = model.fetch(key)
            got = actual.fetch_cached(key)
            if expected is None:
                check("random-fetch-miss", (case, op), None, got)
            else:
                arr_eq("random-fetch-hit", expected, got)
        check("random-cache-bytes", (case, op), model.used, actual.bytes_used)
        check(
            "random-cache-entries", (case, op), len(model.items), actual.entry_count()
        )
# Largest practical cache entry and C-contiguous-copy guarantee.
large = BudgetArrayCache(4 * 1024 * 1024)
data = np.arange(262144, dtype=np.float32).reshape(512, 512)
check("largest-cache-insert", 0, True, large.store_cached("large", data))
check("largest-cache-bytes", 0, data.nbytes, large.bytes_used)
arr_eq("largest-cache-fetch", data, large.fetch_cached("large"))
