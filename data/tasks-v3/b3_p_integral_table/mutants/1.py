import numpy as np
def integral_table_2d(x):
 a=np.asarray(x,dtype=float)
 if a.ndim!=2: raise ValueError()
 return a.cumsum(0).cumsum(1)
