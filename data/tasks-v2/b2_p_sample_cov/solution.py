import numpy as np
def sample_covariance(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] < 2 or not np.isfinite(a).all():
        raise ValueError("expected at least two finite rows")
    centered = a - np.mean(a, axis=0)
    return (centered.T @ centered) / (a.shape[0] - 1)
