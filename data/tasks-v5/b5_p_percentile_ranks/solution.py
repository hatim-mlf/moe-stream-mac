import numpy as np

def percentile_ranks(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError
    n=len(a)
    if not n: return np.empty(0,dtype=np.float64)
    s=np.sort(a)
    lo=np.searchsorted(s,a,side='left'); hi=np.searchsorted(s,a,side='right')
    return (lo+hi)/(2.0*n)
