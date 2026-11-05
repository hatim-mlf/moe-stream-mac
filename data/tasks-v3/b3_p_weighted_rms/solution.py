import numpy as np

def weighted_rms(values, weights):
    x=np.asarray(values,dtype=np.float64); w=np.asarray(weights,dtype=np.float64)
    if x.ndim!=1 or w.ndim!=1 or x.size==0 or x.shape!=w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0: raise ValueError("invalid values or weights")
    w=w/np.max(w)
    denom=np.sum(w)
    scale=np.max(np.abs(x))
    if scale==0: return 0.0
    return float(scale*np.sqrt(np.sum(w*(x/scale)**2)/denom))
