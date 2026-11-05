import numpy as np
def rgb_luma(rgb):
    a=np.asarray(rgb,dtype=np.float64)
    if a.ndim==0 or a.shape[-1]!=3:
        raise ValueError("last dimension must contain RGB")
    return a[...,0]*0.2126+a[...,1]*0.7152+a[...,2]*0.0722
