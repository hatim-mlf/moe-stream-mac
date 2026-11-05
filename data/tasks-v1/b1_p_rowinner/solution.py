import numpy as np
def rowwise_inner(left,right):
    a=np.asarray(left)
    b=np.asarray(right)
    if a.ndim!=2 or b.ndim!=2 or a.shape!=b.shape:
        raise ValueError("expected equal matrices")
    return np.sum(a.astype(np.float64)*b.astype(np.float64),axis=1,dtype=np.float64)
