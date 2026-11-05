Target: candidate.py (Python 3.14; standard library and NumPy only)

Sum leading-axis rows into a fixed number of segments with explicit empty-segment behavior.

Exact public signature:
```python
def segment_reduce_sum(values, segment_ids, n_segments): ...
```

`values` has rank >=1. `segment_ids` is a rank-1 integer array with one id per leading row. `n_segments` is an integer (not bool) >=0, and every id is in [0,n_segments). Invalid shape or bounds raises ValueError; invalid dtype raises TypeError. Return shape `(n_segments,) + values.shape[1:]`, with float64 dtype, summing rows into their segment. Empty segments are all zeros; an empty values input is valid only when ids is empty. Inputs are not modified.

Return only the complete file in one fenced python block.
