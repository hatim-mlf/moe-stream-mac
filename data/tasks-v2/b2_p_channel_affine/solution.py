import numpy as np
def channel_affine(x, scale, bias):
    a = np.asarray(x, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    b = np.asarray(bias, dtype=np.float64)
    if a.ndim < 1 or not np.isfinite(a).all(): raise ValueError("invalid input")
    channels = a.shape[-1]
    for p in (s, b):
        if p.ndim > 1 or (p.ndim == 1 and p.size != channels) or not np.isfinite(p).all():
            raise ValueError("invalid channel parameter")
    return a * s + b
