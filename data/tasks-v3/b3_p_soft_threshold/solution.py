import numpy as np

def soft_threshold_array(x, threshold):
    a=np.asarray(x,dtype=np.float64)
    if not np.isfinite(a).all() or not np.isfinite(threshold) or threshold<0: raise ValueError("invalid values")
    return np.sign(a)*np.maximum(np.abs(a)-threshold,0.0)
