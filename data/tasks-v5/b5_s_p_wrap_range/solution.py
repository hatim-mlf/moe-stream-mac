import numpy as np

def wrap_to_range(values,low,high):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim<0 or not np.isfinite(a).all() or not np.isscalar(low) or not np.isscalar(high) or not np.isfinite(low) or not np.isfinite(high) or high<=low: raise ValueError
    return ((a-float(low))%(float(high)-float(low))+float(low)).astype(np.float64)
