import numpy as np

def reflect_pad_1d(values,left,right):
    a=np.asarray(values)
    if a.ndim!=1 or left<0 or right<0: raise ValueError
    if a.size==0 and (left or right): raise ValueError
    return np.pad(a,(left,right),mode='reflect')
