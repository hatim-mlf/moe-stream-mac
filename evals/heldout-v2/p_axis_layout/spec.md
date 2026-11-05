Target: candidate.py (Python 3.14; standard library and NumPy only)

Move one feature axis to a destination while returning a contiguous copy.

Exact public signature:
```python
def move_feature_axis(values, axis, destination): ...
```

Convert values with NumPy and require rank >=1. Both axes must be integers but not bool; normalize negatives against ndim and require both in [0,ndim). Invalid rank/range raises ValueError; invalid axis type raises TypeError. Move the source axis to the destination using NumPy axis-move semantics. Return a C-contiguous independent copy with the same dtype; do not mutate input.

Return only the complete file in one fenced python block.
