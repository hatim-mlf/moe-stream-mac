import numpy as np
def column_zscore(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] == 0 or not np.isfinite(a).all():
        raise ValueError("expected nonempty finite matrix")
    mean = np.mean(a, axis=0)
    std = np.std(a, axis=0, ddof=0)
    out = np.zeros_like(a)
    np.divide(a - mean, std, out=out, where=std != 0)
    return out
