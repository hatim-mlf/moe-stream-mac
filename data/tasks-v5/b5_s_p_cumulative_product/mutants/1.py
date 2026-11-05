import numpy as np

def cumulative_product(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1: raise ValueError
    return np.cumprod(a[::-1])[::-1]
