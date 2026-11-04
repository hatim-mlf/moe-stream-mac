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
