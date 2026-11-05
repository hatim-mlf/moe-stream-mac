import numpy as np
def split_by_gap(values,max_gap):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or not np.all(np.isfinite(a)) or np.any(np.diff(a)<0): raise ValueError()
    if not np.isfinite(max_gap) or max_gap<0: raise ValueError()
    if not len(a): return []
    starts=[0]
    for i in range(1,len(a)):
        if a[i]-a[i-1]>=max_gap: starts.append(i)
    starts.append(len(a))
    return [a[starts[i]:starts[i+1]].copy() for i in range(len(starts)-1)]
