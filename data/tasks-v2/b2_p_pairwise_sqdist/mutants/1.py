import numpy as np
def pairwise_sqdist(x,y):
    a=np.asarray(x,dtype=np.float64); b=np.asarray(y,dtype=np.float64)
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]: raise ValueError()
    d=a[:,None,:]-b[None,:,:]
    return np.sqrt(np.sum(d*d,axis=2,dtype=np.float64))
