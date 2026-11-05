import numpy as np

def derivative_nonuniform(y, x):
    v = np.asarray(y, dtype=np.float64)
    t = np.asarray(x, dtype=np.float64)
    if v.ndim != 1 or t.ndim != 1 or v.size != t.size or v.size < 2:
        raise ValueError('expected equal vectors of length at least two')
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('values must be finite and x increasing')
    h = np.diff(t)
    slopes = np.diff(v) / h
    out = np.empty_like(v)
    out[0], out[-1] = slopes[0], slopes[-1]
    if v.size > 2:
        out[1:-1] = (h[1:] * slopes[:-1] + h[:-1] * slopes[1:]) / (h[:-1] + h[1:])
    return out
