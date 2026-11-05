import numpy as np

def signed_polygon_area(points):
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all(): raise ValueError("expected finite points")
    if len(p)<3: return 0.0
    area=np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))
    return float(area*0.5)
