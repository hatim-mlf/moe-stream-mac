Target: candidate.py (Python 3.14; standard library and NumPy only)

Create copied sliding windows along the leading axis.

Exact public signature:
```python
def window_matrix_copy(values, width, step): ...
```

Accept a NumPy-convertible array of rank >=1. `width` and `step` must be integers but not bool; require width>0, step>0, and width<=values.shape[0]. Invalid rank/range raises ValueError; invalid parameter types raise TypeError. Window starts are `0, step, 2*step, ...` while a full width fits. Return a fresh array with shape `(number_of_windows,width)+values.shape[1:]` and the same dtype. Mutating the result must not mutate the input.

Return only the complete file in one fenced python block.
