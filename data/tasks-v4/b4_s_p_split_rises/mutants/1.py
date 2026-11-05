import numpy as np
def split_nondecreasing_runs(values):
    a=np.asarray(values)
    if a.ndim!=1: raise ValueError()
    if not len(a): return []
    starts=[0]
    for i in range(1,len(a)):
        if a[i]<=a[i-1]: starts.append(i)
    starts.append(len(a))
    return [a[starts[i]:starts[i+1]].copy() for i in range(len(starts)-1)]
