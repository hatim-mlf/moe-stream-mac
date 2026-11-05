import numpy as np
def smoothstep01(x):
 a=np.asarray(x,dtype=float)
 if not np.isfinite(a).all(): raise ValueError()
 return np.clip(a,0,1)**2*(3-2*a)
