import numpy as np

def integral_table_2d(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError("expected finite matrix")
    out=np.zeros((a.shape[0]+1,a.shape[1]+1),dtype=np.float64)
    out[1:,1:]=a.cumsum(axis=0).cumsum(axis=1)
    return out
