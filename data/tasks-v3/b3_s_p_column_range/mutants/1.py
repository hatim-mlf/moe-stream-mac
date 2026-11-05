import numpy as np
def column_range(x):
 a=np.asarray(x,dtype=float)
 return np.max(a,axis=0)
