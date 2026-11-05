import numpy as np
def diagonal_band_mask(rows, cols, offset, radius):
    vals = (rows, cols, offset, radius)
    if any(not isinstance(v, (int, np.integer)) or isinstance(v, (bool, np.bool_)) for v in vals):
        raise ValueError("dimensions and offsets must be integers")
    if rows < 0 or cols < 0 or radius < 0:
        raise ValueError("negative dimension or radius")
    r = np.arange(rows)[:, None]
    c = np.arange(cols)[None, :]
    return np.abs((c - r) - offset) <= radius
