import numpy as np

def wrap_to_range(values,low,high):
    a=np.asarray(values,dtype=float)
    return np.clip(a,low,high)
