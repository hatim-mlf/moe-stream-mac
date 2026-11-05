import numpy as np

def rowwise_argmin(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]==0 or not np.isfinite(a).all(): raise ValueError
    return np.argmin(a,axis=1).astype(np.int64)
