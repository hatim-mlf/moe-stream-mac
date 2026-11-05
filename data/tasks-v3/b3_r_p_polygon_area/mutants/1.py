import numpy as np
def signed_polygon_area(points):
 p=np.asarray(points,dtype=float)
 if p.ndim!=2 or p.shape[1]!=2 or not np.isfinite(p).all(): raise ValueError()
 if len(p)<3:return 0.0
 return abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))*0.5))
