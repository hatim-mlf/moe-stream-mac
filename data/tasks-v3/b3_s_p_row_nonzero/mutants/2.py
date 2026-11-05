import numpy as np
def rows_with_nonzero(x):
 a=np.asarray(x)
 if a.ndim!=2: raise ValueError()
 return np.flatnonzero(np.all(a!=0,axis=1)).astype(np.int64)
