import numpy as np
def weighted_median(values,weights):
    v=np.asarray(values,dtype=np.float64)
    w=np.asarray(weights,dtype=np.float64)
    if v.ndim!=1 or w.ndim!=1 or v.size==0 or v.shape!=w.shape:
        raise ValueError("expected matching nonempty vectors")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(w)) or np.any(w<0) or np.sum(w)<=0:
        raise ValueError("invalid values or weights")
    order=np.argsort(v,kind="stable")
    sorted_w=w[order]
    index=int(np.searchsorted(np.cumsum(sorted_w),np.sum(sorted_w)/2.0,side="left"))
    return float(v[order[index]])
