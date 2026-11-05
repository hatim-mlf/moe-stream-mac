import numpy as np

def boundary_pixels_4(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or a.dtype!=np.bool_: raise ValueError
    up=np.zeros_like(a); down=np.zeros_like(a); left=np.zeros_like(a); right=np.zeros_like(a)
    up[1:]=a[:-1]; down[:-1]=a[1:]; left[:,1:]=a[:,:-1]; right[:,:-1]=a[:,1:]
    return a & ~(up & down & left & right)
