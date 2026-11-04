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


## Your previous attempt failed

```python
import numpy as np
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Tuple, Optional, Any

class PredictivePrefetcher:
    def __init__(self, reader, projs: tuple):
        self.reader = reader
        self.projs = projs
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending: Dict[int, Tuple[np.ndarray, Future]] = {}
        self.predicted_hits = 0
        self.misses = 0

    def request(self, layer: int, predicted: np.ndarray) -> None:
        if layer in self._pending or len(predicted) == 0:
            return
        ids = np.unique(predicted)
        future = self._executor.submit(self._fetch_layer, layer, ids)
        self._pending[layer] = (ids, future)

    def take(self, layer: int, needed: np.ndarray) -> dict:
        needed_ids = np.unique(needed)
        result: Dict[int, Dict[str, Any]] = {}
        if layer in self._pending:
            ids, future = self._pending[layer]
            future.result()  # Re-raise errors if any
            for nid in needed_ids:
                idx = np.searchsorted(ids, nid)
                if idx < len(ids) and ids[idx] == nid:
                    self.predicted_hits += 1
                    result[nid] = self._extract_row(ids, nid, self._pending[layer][1])
                else:
                    self.misses += 1
                    result[nid] = self._read_immediate(layer, nid)
            del self._pending[layer]
        else:
            for nid in needed_ids:
                self.misses += 1
                result[nid] = self._read_immediate(layer, nid)
        return result

    def stats(self) -> dict:
        total = self.predicted_hits + self.misses
        return {
            "predicted_hits": self.predicted_hits,
            "misses": self.misses,
            "hit_rate": self.predicted_hits / total if total > 0 else 0.0
        }

    def _fetch_layer(self, layer: int, ids: np.ndarray) -> dict:
        return {proj: self.reader.read_sorted(layer, proj, ids) for proj in self.projs}

    def _extract_row(self, ids: np.ndarray, nid: int, future: Future) -> Dict[str, Any]:
        data = future.result()
        idx = np.searchsorted(ids, nid)
        return {proj: {k: arr[idx] for k, arr in proj_data.items()} for proj, proj_data in data.items()}

    def _read_immediate(self, layer: int, nid: int) -> Dict[str, Any]:
        return {proj: self.reader.read_sorted(layer, proj, np.array([nid]))[proj] for proj in self.projs}
```

Test failure: KeyError: gate_proj in _read_immediate.

Code review (fix all four, keep request/stats/_fetch_layer as they are):
1. read_sorted returns {"weight": arr, "scales": arr, "biases": arr} - there is no [proj] key. A row is
   (arr["weight"][i], arr["scales"][i], arr["biases"][i]).
2. Every value must be a TUPLE (weight_row, scales_row, biases_row) per proj, not a dict:
   result[e][proj] = (data[proj]["weight"][i], data[proj]["scales"][i], data[proj]["biases"][i])
3. Misses: collect ALL uncovered ids first, then ONE read_sorted(layer, proj, np.array(sorted_misses)) per proj,
   and fill result from its rows. Delete _read_immediate.
4. Result keys must be Python ints: int(e).
Call future.result() once and keep the data in a local variable; delete _extract_row.


Fix it. Reply with the complete corrected file.