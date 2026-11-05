import numpy as np

def cumulative_product(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError
    return np.cumprod(a,dtype=np.float64)
