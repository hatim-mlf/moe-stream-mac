import numpy as np

def first_unique_rows(x):
    a = np.asarray(x)
    if a.ndim != 2 or a.dtype.kind not in 'biuf':
        raise ValueError('expected numeric matrix')
    if not np.all(np.isfinite(a)):
        raise ValueError('values must be finite')
    keep = []
    for i, row in enumerate(a):
        if not any(np.array_equal(row, a[j]) for j in keep):
            keep.append(i)
    return np.asarray(keep, dtype=np.int64)
