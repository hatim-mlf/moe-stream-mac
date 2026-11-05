import numpy as np

def conv2d_valid(image, kernel):
    x = np.asarray(image, dtype=np.float64)
    k = np.asarray(kernel, dtype=np.float64)
    if x.ndim != 2 or k.ndim != 2 or min(k.shape, default=0) <= 0:
        raise ValueError('expected nonempty 2-D kernel')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(k)):
        raise ValueError('values must be finite')
    h, w = max(x.shape[0]-k.shape[0]+1, 0), max(x.shape[1]-k.shape[1]+1, 0)
    out = np.empty((h, w), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            out[r, c] = np.sum(x[r:r+k.shape[0], c:c+k.shape[1]] * k)
    return out
