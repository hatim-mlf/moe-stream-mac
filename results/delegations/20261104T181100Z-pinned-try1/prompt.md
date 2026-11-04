# Task: write engine/hybrid/pinned.py (Python 3.14, numpy only)

Keep the most-used MoE experts permanently in memory. Two parts: a pure selection function and a loader.

The reader (exists; do not write it):
```python
reader.read_sorted(layer: int, proj: str, experts: np.ndarray) -> dict
    # experts STRICTLY INCREASING. Returns {"weight": arr, "scales": arr, "biases": arr}; row i belongs to experts[i].
```

Write exactly:

```python
def select_pinned(counts: np.ndarray, expert_bytes: int, budget_bytes: int) -> dict:
    """counts: int64 [layers, experts] usage counts. Rank all (layer, expert) pairs by count DESCENDING, ties by
    lower layer then lower expert (use np.lexsort). Skip pairs with count 0. Take pairs in that order while
    (number taken + 1) * expert_bytes <= budget_bytes. Return {layer: np.ndarray of chosen expert ids, sorted
    ascending, dtype int64}; layers with nothing chosen are absent (so zero budget -> {})."""

class PinnedExperts:
    def __init__(self, reader, projs: tuple, selection: dict):
        """For each layer in selection: for each proj, call reader.read_sorted(layer, proj, ids) ONCE, and store
        rows in self._rows[(layer, int(e))] = {proj: (weight_row, scales_row, biases_row)} (copies:
        np.ascontiguousarray of each row, so the big batch arrays can be freed)."""

    def get(self, layer: int, expert: int):
        """The stored dict for (layer, expert), or None."""

    def __len__(self) -> int: ...

    @property
    def nbytes(self) -> int:
        """Sum of .nbytes of every stored array."""
```
Rules: import numpy only. Each function under 30 lines. Return only the complete file in one fenced python block.
