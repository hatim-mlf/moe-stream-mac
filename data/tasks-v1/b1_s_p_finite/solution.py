import numpy as np
def finite_counts_2d(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=2:
        raise ValueError("expected matrix")
    return np.count_nonzero(np.isfinite(a),axis=1).astype(np.int64)
