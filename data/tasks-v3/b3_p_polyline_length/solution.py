import numpy as np

def polyline_length2d(points):
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all(): raise ValueError("expected finite 2-D points")
    return float(np.linalg.norm(np.diff(p,axis=0),axis=1).sum())
