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
    if np.any(p>=a.shape[ax]):
        raise IndexError("position out of range")
    return np.take(a,p.astype(np.intp,copy=False),axis=ax).copy()
