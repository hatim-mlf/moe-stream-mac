import numpy as np

def fill_nearest_1d(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or np.isinf(a).any(): raise ValueError
    good=np.flatnonzero(np.isfinite(a))
    if not len(good):
        if not len(a): return a.copy()
        raise ValueError
    pos=np.arange(len(a)); r=np.searchsorted(good,pos); ri=np.minimum(r,len(good)-1); li=np.maximum(r-1,0)
    choose_r=(np.abs(good[ri]-pos)<np.abs(good[li]-pos))
    take=np.where(choose_r,ri,li)
    return a[good[take]].copy()
