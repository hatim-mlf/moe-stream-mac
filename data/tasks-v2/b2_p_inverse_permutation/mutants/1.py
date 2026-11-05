import numpy as np
def inverse_permutation(indices):
 a=np.asarray(indices)
 if a.ndim!=1 or (a.size and not np.issubdtype(a.dtype,np.integer)): raise ValueError()
 if np.any(a<0) or np.any(a>=a.size) or np.unique(a).size!=a.size: raise ValueError()
 return a.astype(np.int64)
