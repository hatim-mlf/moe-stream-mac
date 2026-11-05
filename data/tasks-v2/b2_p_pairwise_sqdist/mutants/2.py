import numpy as np
def pairwise_sqdist(x,y):
    a=np.asarray(x,dtype=np.float64); b=np.asarray(y,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1] or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError()
    d=a[:,None,:]-b[None,:,:]
    return np.sum(d*d,axis=1,dtype=np.float64)
