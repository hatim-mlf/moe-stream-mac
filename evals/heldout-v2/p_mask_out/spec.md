Target: candidate.py (Python 3.14; standard library and NumPy only)

Scale selected floating values and return an independent array.

Exact public signature:
```python
def masked_scale_copy(values, mask, factor): ...
```

Values must be a NumPy-convertible floating array (float16/32/64); mask must be boolean and broadcastable to values. Factor must be a finite real scalar. Raise TypeError for dtype mismatch and ValueError for invalid factor or broadcasting. Return a fresh array with exactly values.dtype: selected elements are multiplied by factor, others copied unchanged. Do not mutate inputs.

Return only the complete file in one fenced python block.
