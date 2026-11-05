import numpy as np

def merge_sorted_runs(runs):
    arrays=[np.asarray(run) for run in runs]
    dtype=None
    for a in arrays:
        if a.ndim!=1 or a.dtype.kind not in "iuf": raise TypeError("runs must be one-dimensional real numeric arrays")
        if dtype is None: dtype=a.dtype
        elif a.dtype!=dtype: raise TypeError("all runs must have the same dtype")
        if a.size>1 and np.any(a[1:]<a[:-1]): raise ValueError("run is not sorted")
    if dtype is None: dtype=np.dtype(np.float64)
    values=np.concatenate(arrays) if arrays else np.empty(0,dtype=dtype)
    origins=np.concatenate([np.full(a.size,i,dtype=np.int64) for i,a in enumerate(arrays)]) if arrays else np.empty(0,dtype=np.int64)
    order=np.lexsort((np.arange(values.size,dtype=np.int64),values,origins))
    return values[order].copy(),origins[order].copy()
