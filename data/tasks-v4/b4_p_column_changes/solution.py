import numpy as np

def column_change_counts(x):
    a = np.asarray(x)
    if a.ndim != 2:
        raise ValueError('expected a matrix')
    if a.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.int64)
    return np.count_nonzero(a[1:] != a[:-1], axis=0).astype(np.int64)
