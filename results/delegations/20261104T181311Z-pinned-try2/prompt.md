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


## Your previous attempt failed

```python
import numpy as np

def select_pinned(counts: np.ndarray, expert_bytes: int, budget_bytes: int) -> dict:
    """Select top experts by count within byte budget."""
    if budget_bytes <= 0 or expert_bytes <= 0:
        return {}
    layers, experts = np.where(counts > 0)
    if layers.size == 0:
        return {}
    order = np.lexsort((experts, layers), axis=0)
    sorted_counts = counts[layers[order], experts[order]]
    sorted_layers = layers[order]
    sorted_experts = experts[order]
    chosen = {}
    count = 0
    for i in range(len(sorted_counts)):
        if (count + 1) * expert_bytes > budget_bytes:
            break
        l, e = int(sorted_layers[i]), int(sorted_experts[i])
        if l not in chosen:
            chosen[l] = []
        chosen[l].append(e)
        count += 1
    return {l: np.array(sorted_experts_list, dtype=np.int64) for l, sorted_experts_list in chosen.items()}


class PinnedExperts:
    def __init__(self, reader, projs: tuple, selection: dict):
        """Load pinned experts from reader."""
        self._rows = {}
        for layer, expert_ids in selection.items():
            for proj in projs:
                result = reader.read_sorted(layer, proj, expert_ids)
                for i, expert_id in enumerate(expert_ids):
                    key = (layer, int(expert_id))
                    if key not in self._rows:
                        self._rows[key] = {}
                    self._rows[key][proj] = (
                        np.ascontiguousarray(result["weight"][i]),
                        np.ascontiguousarray(result["scales"][i]),
                        np.ascontiguousarray(result["biases"][i]),
                    )

    def get(self, layer: int, expert: int):
        """Get stored data for (layer, expert), or None."""
        return self._rows.get((layer, int(expert)))

    def __len__(self) -> int:
        """Number of pinned experts."""
        return len(self._rows)

    @property
    def nbytes(self) -> int:
        """Sum of .nbytes of every stored array."""
        total = 0
        for key in self._rows:
            for proj in self._rows[key]:
                w, s, b = self._rows[key][proj]
                total += w.nbytes + s.nbytes + b.nbytes
        return total
```

Test: counts (0,1)=50, (1,0)=50, (1,5)=30, (2,2)=40, (0,4)=10, budget 3 experts. Expected {0:[1], 1:[0], 2:[2]};
yours returned {0:[1,4], 1:[0]}.

Code review (select_pinned only; PinnedExperts is accepted, keep it unchanged):
1. The ranking ignores the counts. np.lexsort sorts by its LAST key first, so use
     vals = counts[layers, experts]; order = np.lexsort((experts, layers, -vals))
   (count descending, then lower layer, then lower expert).
2. Each layer's chosen ids must be sorted ascending: np.array(sorted(ids), dtype=np.int64).
Return the whole file.


Fix it. Reply with the complete corrected file.