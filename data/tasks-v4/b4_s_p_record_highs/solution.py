import numpy as np

def record_high_indices(values):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'biuf' or not np.all(np.isfinite(a)):
        raise ValueError('expected finite numeric vector')
    keep = []
    best = None
    for i, value in enumerate(a):
        if i == 0 or value > best:
            keep.append(i)
            best = value
    return np.asarray(keep, dtype=np.int64)
