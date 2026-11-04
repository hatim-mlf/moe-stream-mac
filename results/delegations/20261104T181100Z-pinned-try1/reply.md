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