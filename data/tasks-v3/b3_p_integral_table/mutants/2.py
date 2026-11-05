import numpy as np
def integral_table_2d(x):
 a=np.asarray(x,dtype=float)
 if a.ndim!=2 or not np.isfinite(a).all(): raise ValueError()
 o=np.zeros((a.shape[0]+1,a.shape[1]+1)); o[1:,1:]=a.cumsum(1)
 return o
