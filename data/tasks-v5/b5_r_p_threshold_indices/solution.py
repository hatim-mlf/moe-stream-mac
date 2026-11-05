import numpy as np

def indices_above_threshold(scores,threshold):
    a=np.asarray(scores,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all() or not np.isscalar(threshold) or isinstance(threshold,(bool,np.bool_)) or not np.isfinite(threshold): raise ValueError
    return np.flatnonzero(a>float(threshold)).astype(np.int64)
