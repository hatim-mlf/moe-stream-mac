import numpy as np

def rowwise_argmin(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or a.shape[1]==0 or not np.isfinite(a).all(): raise ValueError
    return (a.shape[1]-1-np.argmin(a[:,::-1],axis=1)).astype(np.int64)
