Target: candidate.py (Python 3.14; standard library and NumPy only)

Merge sorted one-dimensional NumPy runs with stable run-order ties.

Exact public signature:
```python
def merge_sorted_runs(runs): ...
```

`runs` is an iterable of rank-1 arrays, all with the same real numeric dtype and each sorted nondecreasing. Wrong rank/dtype or mismatched dtypes raises TypeError; unsorted values raise ValueError. Return `(values, origins)`: merged values preserve the common dtype, origins is int64 with each source run index. Equal values are ordered by lower run index, preserving order within each run. Empty input returns empty float64 values and empty int64 origins. Return independent arrays and do not mutate inputs.

Return only the complete file in one fenced python block.
