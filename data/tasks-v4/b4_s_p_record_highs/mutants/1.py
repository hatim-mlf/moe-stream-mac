import numpy as np
def record_high_indices(values):
    a=np.asarray(values)
    if a.ndim!=1: raise ValueError()
    keep=[]; best=None
    for i,v in enumerate(a):
        if i==0 or v>=best: keep.append(i); best=v
    return np.asarray(keep,dtype=np.int64)
