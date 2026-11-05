import numpy as np

def rotate_points2d(points, angle):
    p=np.asarray(points,dtype=np.float64)
    if p.ndim<1 or p.shape[-1]!=2 or not np.isfinite(p).all() or not np.isfinite(angle): raise ValueError("invalid points")
    c=np.cos(angle); s=np.sin(angle)
    return np.stack((c*p[...,0]-s*p[...,1],s*p[...,0]+c*p[...,1]),axis=-1)
