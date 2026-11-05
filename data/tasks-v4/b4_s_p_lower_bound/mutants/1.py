import numpy as np
def lower_bound_index(values,target):
    a=np.asarray(values,dtype=float)
    if a.ndim!=1 or np.any(np.diff(a)<0): raise ValueError()
    lo,hi=0,len(a)
    while lo<hi:
        m=(lo+hi)//2
        if a[m]<=target: lo=m+1
        else: hi=m
    return lo
