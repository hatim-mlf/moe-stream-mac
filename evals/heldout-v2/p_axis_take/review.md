Target: candidate.py (Python 3.14; standard library and NumPy only)

Gather positions along one normalized axis, preserving dtype and returning independent storage.

Exact public signature:
```python
def take_axis_positions(array, positions, axis): ...
```

Convert array-like inputs with NumPy. Reject a scalar array with ValueError. `axis` must be an integer but not bool; normalize negative axes and raise ValueError when out of range. `positions` must be rank-1 integer dtype (signed or unsigned, not bool); otherwise raise TypeError. Every index must satisfy `0 <= i < array.shape[axis]`, otherwise raise IndexError (negative indices are not accepted). Return a copy with that axis replaced by len(positions), preserving array dtype.

Return only the complete file in one fenced python block.

## Your previous attempt

```python
import numpy as np

def take_axis_positions(array, positions, axis):
    a=np.asarray(array)
    p=np.asarray(positions)
    if a.ndim==0:
        raise ValueError("array must have at least one dimension")
    if not isinstance(axis,(int,np.integer)) or isinstance(axis,(bool,np.bool_)):
        raise TypeError("axis must be an integer")
    ax=int(axis)
    if ax<0: ax+=a.ndim
    if ax<0 or ax>=a.ndim:
        raise ValueError("axis out of range")
    if p.ndim!=1 or p.dtype.kind not in "iu":
        raise TypeError("positions must be a one-dimensional integer array")
    if np.any(p<0) or np.any(p>=a.shape[ax]):
        raise IndexError("position out of range")
    return np.take(a,p.astype(np.intp,copy=False),axis=0).copy()
```

## Review notes

Use the normalized requested axis for gathering and reject negative positions rather than inheriting NumPy wraparound.
