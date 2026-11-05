Target: candidate.py (Python 3.14; standard library and NumPy only)

Implement a least-recently-used NumPy array cache constrained by stored byte size.

Exact public signature:
```python
class BudgetArrayCache:
    def __init__(self, budget_bytes): ...
    def entry_count(self): ...
    @property
    def bytes_used(self): ...
    def fetch_cached(self, key): ...
    def store_cached(self, key, array): ...
```

Budget must be a nonnegative Python int but not bool; invalid values raise ValueError. Store copies as C-contiguous arrays. `store_cached` requires a hashable key and an ndarray (otherwise TypeError). Replacing a key removes its old bytes first. If the new array exceeds the whole budget, do not cache it and return False. Otherwise evict least-recently-used entries until it fits, insert as most-recently-used, and return True. `fetch_cached` returns None on miss; on hit it marks the entry most-recently-used and returns a copy. `bytes_used` counts stored arrays only and `entry_count` counts entries.

Return only the complete file in one fenced python block.
