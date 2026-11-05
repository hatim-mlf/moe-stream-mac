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
        return value.copy()

    def store_cached(self,key,array):
        try: hash(key)
        except TypeError as exc: raise TypeError("key must be hashable") from exc
        if not isinstance(array,np.ndarray): raise TypeError("array must be a NumPy array")
        stored=np.ascontiguousarray(array).copy()
        if key in self._items: self._nbytes-=self._items.pop(key).nbytes
        if stored.nbytes>self.budget_bytes: return False
        while self._items and self._nbytes+stored.nbytes>self.budget_bytes:
            _,old=self._items.popitem(last=(len(self._items)>=3)); self._nbytes-=old.nbytes
        self._items[key]=stored; self._nbytes+=stored.nbytes
        return True
