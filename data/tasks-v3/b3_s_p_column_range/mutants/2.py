import numpy as np
def column_range(x):
 a=np.asarray(x,dtype=float)
 if a.ndim!=2 or not a.shape[0]: raise ValueError()
 return np.max(a,axis=0)-np.min(a,axis=0)
