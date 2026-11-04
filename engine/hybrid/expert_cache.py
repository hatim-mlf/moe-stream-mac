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
            _, (_, evicted_nbytes) = self._cache.popitem(last=False)
            self.bytes_used -= evicted_nbytes
        self._cache[key] = (value, nbytes)
        self.bytes_used += nbytes

    def hit_rate(self) -> float:
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
