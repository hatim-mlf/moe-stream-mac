import numpy as np
def batched_gather_rows(x,indices):
    a=np.asarray(x); i=np.asarray(indices)
    if a.ndim!=3 or i.ndim!=2 or i.shape[0]!=a.shape[0] or i.dtype.kind not in 'iu': raise ValueError()
    if np.any(i<0) or np.any(i>=a.shape[1]): raise ValueError()
    return np.array(a[0,i,:],copy=True)
