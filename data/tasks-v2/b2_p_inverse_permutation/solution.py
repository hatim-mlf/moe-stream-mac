import numpy as np
def inverse_permutation(indices):
    a = np.asarray(indices)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError("expected integer vector")
    n = a.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if np.any(a < 0) or np.any(a >= n) or np.unique(a).size != n:
        raise ValueError("indices must be a permutation")
    out = np.empty(n, dtype=np.int64)
    out[a.astype(np.intp)] = np.arange(n, dtype=np.int64)
    return out
