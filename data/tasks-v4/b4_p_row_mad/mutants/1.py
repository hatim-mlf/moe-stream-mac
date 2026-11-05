import numpy as np
def row_mad(x,axis=0):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim<1: raise ValueError()
    return np.asarray(np.mean(np.abs(a-np.mean(a,axis=axis,keepdims=True)),axis=axis),dtype=np.float64)
