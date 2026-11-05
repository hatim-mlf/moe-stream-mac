import numpy as np
def weighted_median(values,weights):
    v=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
    order=np.argsort(v)
    return float(v[order[np.searchsorted(np.cumsum(w[order]),w.sum()/2,side="right")]])
