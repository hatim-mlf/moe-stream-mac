import numpy as np

def minmax_scale_columns(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError
    if a.shape[0]==0:return np.empty(a.shape,dtype=np.float64)
    lo=a.min(axis=0,keepdims=True); span=a.max(axis=0,keepdims=True)-lo
    return np.divide(a-lo,span,out=np.zeros_like(a),where=span!=0)
