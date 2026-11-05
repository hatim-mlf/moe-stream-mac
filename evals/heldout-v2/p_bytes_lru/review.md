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

## Your previous attempt

```python
from collections import OrderedDict
import numpy as np

class BudgetArrayCache:
    def __init__(self, budget_bytes):
        if isinstance(budget_bytes,bool) or not isinstance(budget_bytes,int) or budget_bytes<0:
            raise ValueError("budget_bytes must be a nonnegative integer")
        self.budget_bytes=budget_bytes
        self._items=OrderedDict()
        self._nbytes=0

    def entry_count(self):
        return len(self._items)

    @property
    def bytes_used(self):
        return self._nbytes

    def fetch_cached(self,key):
        if key not in self._items: return None
        value=self._items.pop(key); self._items[key]=value
        return value

    def store_cached(self,key,array):
        try: hash(key)
        except TypeError as exc: raise TypeError("key must be hashable") from exc
        if not isinstance(array,np.ndarray): raise TypeError("array must be a NumPy array")
        stored=np.ascontiguousarray(array).copy()
        if key in self._items: self._nbytes-=self._items.pop(key).nbytes
        if stored.nbytes>self.budget_bytes: return False
        while self._items and self._nbytes+stored.nbytes>self.budget_bytes:
            _,old=self._items.popitem(last=False); self._nbytes-=old.nbytes
        self._items[key]=stored; self._nbytes+=stored.nbytes
        return True
```

## Review notes

On a cache hit return an independent copy and update recency. Track bytes using each stored array’s nbytes.
