import numpy as np

def project_simplex_rows(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] == 0 or not np.all(np.isfinite(a)):
        raise ValueError('expected finite matrix with columns')
    out = np.empty_like(a)
    for i, row in enumerate(a):
        u = np.sort(row)[::-1]
        cssv = np.cumsum(u) - 1.0
        active = np.flatnonzero(u - cssv / np.arange(1, row.size+1) > 0)
        theta = cssv[active[-1]] / (active[-1] + 1)
        out[i] = np.maximum(row - theta, 0.0)
    return out
