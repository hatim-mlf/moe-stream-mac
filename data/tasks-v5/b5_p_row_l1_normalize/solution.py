import numpy as np

def row_l1_normalize(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError
    d=np.sum(np.abs(a),axis=1,keepdims=True)
    return np.divide(a,d,out=np.zeros_like(a),where=d!=0)
