# Task: write engine/hybrid/predictive.py (Python 3.14, numpy, standard library)

One class, `PredictivePrefetcher`. While the GPU works on layer L, it reads a PREDICTED set of layer L+1's experts
from disk in a background thread. When layer L+1 runs, it returns the experts actually NEEDED: predicted ones from
the background read, any others read immediately.

The reader (already exists; do not write it):
```python
reader.read_sorted(layer: int, proj: str, experts: np.ndarray) -> dict
    # experts: STRICTLY INCREASING int ids. Returns {"weight": arr, "scales": arr, "biases": arr},
    # each array shaped [len(experts), ...]; row i belongs to experts[i]. Thread-safe.
```

API (exact):
```python
class PredictivePrefetcher:
    def __init__(self, reader, projs: tuple):
        """projs e.g. ("gate_proj", "up_proj", "down_proj"). One ThreadPoolExecutor(max_workers=1);
        self._pending: dict layer -> (sorted predicted ids ndarray, Future). Counters predicted_hits, misses (ints)."""

    def request(self, layer: int, predicted: np.ndarray) -> None:
        """Start reading np.unique(predicted) for every proj in the background (skip if layer already pending or
        the set is empty). The background job returns {proj: read_sorted(layer, proj, ids)}."""

    def take(self, layer: int, needed: np.ndarray) -> dict:
        """Return {expert_id (int): {proj: (weight_row, scales_row, biases_row)}} for every id in np.unique(needed).
        If layer is pending: wait for its job (re-raise errors) and take rows of ids it holds (row index from
        np.searchsorted on the sorted predicted ids). Ids not covered are read now with read_sorted (sorted, once
        per proj). Count predicted_hits / misses per needed id. Remove the layer from _pending."""

    def stats(self) -> dict:
        """{"predicted_hits": int, "misses": int, "hit_rate": float (0.0 when nothing taken)}"""
```
Rules: rows must be the arrays from read_sorted (no re-reading of predicted hits). Imports: numpy and
concurrent.futures only. Each method under 35 lines. Return only the complete file in one fenced python block.
