import numpy as np

def percentile_ranks(values):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError
    if len(a)<2:return np.full(len(a),.5)
    s=np.sort(a); lo=np.searchsorted(s,a,'left'); hi=np.searchsorted(s,a,'right')
    return (lo+hi-1)/(2*(len(a)-1))
