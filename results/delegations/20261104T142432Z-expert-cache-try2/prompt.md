# Task: write engine/hybrid/expert_cache.py (Python 3.14, standard library only)

One class, `ExpertCache`: a least-recently-used cache with a BYTE budget, used to keep hot MoE experts
resident in GPU memory. Keys are tuples like (layer, expert). Values are opaque objects; the caller passes
each value's size in bytes.

```python
class ExpertCache:
    def __init__(self, budget_bytes: int):
        """Use collections.OrderedDict. Attributes: budget_bytes, bytes_used, hits, misses (all int)."""

    def get(self, key):
        """Return the value and mark it most recently used (hits += 1), or None (misses += 1)."""

    def put(self, key, value, nbytes: int) -> None:
        """Insert or replace. If nbytes > budget_bytes: store nothing, evict nothing, return.
        If key exists: remove the old entry first (subtract its size). Then evict least recently used entries
        until bytes_used + nbytes <= budget_bytes, then insert as most recently used and add nbytes."""

    def hit_rate(self) -> float:
        """hits / (hits + misses), or 0.0 when both are 0."""
```
Store (value, nbytes) per key. No other methods, no prints. Each method under 20 lines.
Return only the complete file in one fenced python block.


## Your previous attempt failed

```python
from collections import OrderedDict


class ExpertCache:
    def __init__(self, budget_bytes: int):
        self.budget_bytes = budget_bytes
        self.bytes_used = 0
        self.hits = 0
        self.misses = 0
        self._cache = OrderedDict()

    def get(self, key):
        if key not in self._cache:
            self.misses += 1
            return None
        value, nbytes = self._cache.pop(key)
        self._cache[key] = (value, nbytes)
        self.hits += 1
        return value

    def put(self, key, value, nbytes: int) -> None:
        if nbytes > self.budget_bytes:
            return
        if key in self._cache:
            _, old_nbytes = self._cache.pop(key)
            self.bytes_used -= old_nbytes
        while self.bytes_used + nbytes > self.budget_bytes and self._cache:
            _, evicted_nbytes = self._cache.popitem(last=False)
            self.bytes_used -= evicted_nbytes
        self._cache[key] = (value, nbytes)
        self.bytes_used += nbytes

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
```

Test failure: line 28, TypeError: unsupported operand type(s) for -=: int and tuple.
Review: OrderedDict.popitem returns (key, stored_value) and you store (value, nbytes), so unpack as  _, (_, evicted_nbytes) = self._cache.popitem(last=False). Change nothing else.


Fix it. Reply with the complete corrected file.