import numpy as np
def scatter_add_rows(base, indices, updates):
    a = np.asarray(base, dtype=np.float64)
    idx = np.asarray(indices)
    u = np.asarray(updates, dtype=np.float64)
    if a.ndim != 2 or idx.ndim != 1 or (idx.size and not np.issubdtype(idx.dtype, np.integer)):
        raise ValueError("invalid base or indices")
    if u.shape != (idx.size, a.shape[1]):
        raise ValueError("updates have wrong shape")
    if np.any(idx < 0) or np.any(idx >= a.shape[0]):
        raise ValueError("index out of range")
    if not np.isfinite(a).all() or not np.isfinite(u).all():
        raise ValueError("values must be finite")
    out = a.copy()
    for i, row in zip(idx, u):
        out[int(i)] += row
    return out
