# Task: add two methods to class ExpertReader (engine/hybrid/expert_reader.py)

The class already has (do not repeat it):
```python
self._tensor_info: dict   # (layer, proj, part) -> {"file_name": str, "abs_offset": int, "per_expert_nbytes": int,
                          #   "per_expert_shape": tuple, "np_dtype": np.uint32 | np.uint16, "num_experts": int}
                          # part is "weight", "scales" or "biases"; expert e starts at abs_offset + e * per_expert_nbytes
self._fds: dict           # file_name -> open fd (os.open)
self._executor            # concurrent.futures.ThreadPoolExecutor
```
Module imports available: os, numpy as np (add nothing else).

Write exactly these two methods (indented as class methods), plus at most one private helper method:

```python
    def num_experts(self, layer: int, proj: str) -> int:
        """Number of experts E of (layer, proj) (from the "weight" entry). KeyError if unknown."""

    def read_sorted(self, layer: int, proj: str, experts: np.ndarray) -> dict:
        """Same result as read(): {"weight": arr, "scales": arr, "biases": arr}, each [len(experts), *per_expert_shape],
        experts in the given order. REQUIREMENT: experts is strictly increasing (raise ValueError otherwise).
        Reads in DISK ORDER with few large requests:
          1. Split experts into runs of consecutive ids, e.g. [5,6,7,40,41,200] -> runs (slot 0, ids 5..7),
             (slot 3, ids 40..41), (slot 5, id 200). A run of k experts is ONE contiguous byte range on disk
             (k * per_expert_nbytes bytes at abs_offset + first_id * per_expert_nbytes) and ONE contiguous
             destination out[slot:slot+k].
          2. Cut every run into chunks of at most CHUNK = 32 * 1024 * 1024 bytes.
          3. Build the list of all chunks of all three parts, SORT it by (file_name, file offset), submit each to
             self._executor: os.preadv(fd, [dest_view[a:b]], file_offset) where
             dest_view = memoryview(out[slot:slot+k]).cast("B"); check the returned byte count == b - a
             (raise OSError otherwise).
          4. Wait for every future (re-raise errors), return the dict."""
```
Each method under 40 lines. Return only these methods (and the helper) in one fenced python block, indented
by 4 spaces as they would appear inside the class.
