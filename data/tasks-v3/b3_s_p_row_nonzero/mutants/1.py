import numpy as np
def rows_with_nonzero(x):
 a=np.asarray(x)
 return np.flatnonzero(np.any(a!=0,axis=0)).astype(np.int64)
