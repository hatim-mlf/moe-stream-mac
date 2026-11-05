import numpy as np

def reflect_pad_1d(values,left,right):
    a=np.asarray(values)
    valid=lambda z:isinstance(z,(int,np.integer)) and not isinstance(z,(bool,np.bool_))
    if a.ndim!=1 or not np.issubdtype(a.dtype,np.number) or not valid(left) or not valid(right) or left<0 or right<0: raise ValueError
    if a.size==0 and (left or right): raise ValueError
    return np.pad(a,(int(left),int(right)),mode='symmetric').copy()
