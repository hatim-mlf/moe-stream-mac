import numpy as np
def rgb_luma(rgb):
    a=np.asarray(rgb,dtype=float)
    return a[...,0]*0.299+a[...,1]*0.587+a[...,2]*0.114
