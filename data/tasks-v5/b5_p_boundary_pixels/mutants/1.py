import numpy as np

def boundary_pixels_4(mask):
    a=np.asarray(mask)
    out=np.zeros_like(a)
    if a.shape[0]>2 and a.shape[1]>2:
        center=a[1:-1,1:-1]
        out[1:-1,1:-1]=center & ~(a[:-2,1:-1]&a[2:,1:-1]&a[1:-1,:-2]&a[1:-1,2:])
    return out
