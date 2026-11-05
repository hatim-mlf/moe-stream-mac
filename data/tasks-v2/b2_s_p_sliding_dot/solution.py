import numpy as np
def sliding_dot_valid(signal, kernel):
    x = np.asarray(signal, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if x.ndim != 1 or k.ndim != 1 or k.size == 0:
        raise ValueError("expected 1-D signal and nonempty kernel")
    if not np.isfinite(x).all() or not np.isfinite(k).all():
        raise ValueError("values must be finite")
    if k.size > x.size:
        return np.empty(0, dtype=np.float64)
    return np.array([np.dot(x[i:i+k.size], k) for i in range(x.size-k.size+1)], dtype=np.float64)
