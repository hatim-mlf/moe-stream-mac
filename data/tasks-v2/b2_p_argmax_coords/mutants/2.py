import numpy as np
def argmax_coords(x):
 a=np.asarray(x,dtype=np.float64)
 if a.ndim!=2 or not a.size or not np.isfinite(a).all(): raise ValueError()
 r,c=np.unravel_index(int(np.argmax(a)),a.shape); return r+1,c+1
