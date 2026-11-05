import numpy as np

def row_mad(x, axis=0):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim < 1 or isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)):
        raise ValueError('invalid rank or axis')
    if axis < -a.ndim or axis >= a.ndim:
        raise ValueError('axis out of range')
    axis = int(axis) % a.ndim
    if a.shape[axis] == 0 or not np.all(np.isfinite(a)):
        raise ValueError('empty reduction or nonfinite values')
    center = np.median(a, axis=axis, keepdims=True)
    return np.asarray(np.median(np.abs(a-center), axis=axis), dtype=np.float64)
