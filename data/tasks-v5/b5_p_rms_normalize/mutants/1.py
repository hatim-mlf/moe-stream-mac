import numpy as np

def normalize_by_rms(x,axis=-1,eps=0.0):
    a=np.asarray(x,dtype=float)
    if a.ndim<1 or not np.isfinite(a).all(): raise ValueError
    d=np.sqrt(np.mean((a-a.mean(axis=axis,keepdims=True))**2,axis=axis,keepdims=True)+eps)
    return np.divide(a,d,out=np.zeros_like(a),where=d!=0)
