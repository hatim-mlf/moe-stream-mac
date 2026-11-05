Target: candidate.py (Python 3.14; standard library and NumPy only)

Gather positions along one normalized axis, preserving dtype and returning independent storage.

Exact public signature:
```python
def take_axis_positions(array, positions, axis): ...
```

Convert array-like inputs with NumPy. Reject a scalar array with ValueError. `axis` must be an integer but not bool; normalize negative axes and raise ValueError when out of range. `positions` must be rank-1 integer dtype (signed or unsigned, not bool); otherwise raise TypeError. Every index must satisfy `0 <= i < array.shape[axis]`, otherwise raise IndexError (negative indices are not accepted). Return a copy with that axis replaced by len(positions), preserving array dtype.

Return only the complete file in one fenced python block.
