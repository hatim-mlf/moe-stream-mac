import numpy as np

def column_range(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]==0 or not np.isfinite(a).all(): raise ValueError("expected finite nonempty rows")
    return np.max(a,axis=0)-np.min(a,axis=0)
