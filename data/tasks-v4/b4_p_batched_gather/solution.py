import numpy as np

def batched_gather_rows(x, indices):
    a = np.asarray(x)
    idx = np.asarray(indices)
    if a.ndim != 3 or idx.ndim != 2 or idx.shape[0] != a.shape[0] or idx.dtype.kind not in 'iu':
        raise ValueError('invalid shapes or index dtype')
    if np.any(idx < 0) or np.any(idx >= a.shape[1]):
        raise ValueError('index out of range')
    rows = np.arange(a.shape[0])[:, None]
    return np.array(a[rows, idx, :], copy=True)
