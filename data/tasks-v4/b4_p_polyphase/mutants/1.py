import numpy as np
def polyphase_decompose(values,factor):
    a=np.asarray(values)
    if a.ndim!=1 or factor<=0 or len(a)%factor: raise ValueError()
    return [a[r::factor].copy() for r in range(factor-1,-1,-1)]
