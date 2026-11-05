import numpy as np

def cumulative_logsumexp(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1:raise ValueError
    return np.log(np.cumsum(np.exp(a)))
