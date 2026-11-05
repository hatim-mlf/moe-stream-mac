import numpy as np
def pairwise_sqdist(x, y):
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise ValueError("incompatible matrices")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("values must be finite")
    d = a[:, None, :] - b[None, :, :]
    return np.sum(d * d, axis=2, dtype=np.float64)
