import numpy as np
def nearest_codebook(values, centers):
    x = np.asarray(values, dtype=np.float64)
    c = np.asarray(centers, dtype=np.float64)
    if x.ndim != 1 or c.ndim != 1 or c.size == 0:
        raise ValueError("expected vectors and nonempty centers")
    if not np.isfinite(x).all() or not np.isfinite(c).all():
        raise ValueError("values must be finite")
    return np.argmin(np.abs(x[:, None] - c[None, :]), axis=1).astype(np.int64)
