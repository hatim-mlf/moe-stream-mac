import numpy as np
def dilate_mask_1d(mask, radius):
    a = np.asarray(mask)
    if a.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    if not isinstance(radius, (int, np.integer)) or isinstance(radius, (bool, np.bool_)) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    source = a.astype(bool, copy=False)
    out = np.zeros(source.size, dtype=bool)
    for i in range(source.size):
        lo = max(0, i - int(radius))
        hi = min(source.size, i + int(radius) + 1)
        out[i] = np.any(source[lo:hi])
    return out
