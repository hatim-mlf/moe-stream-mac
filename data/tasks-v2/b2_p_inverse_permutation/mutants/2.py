import numpy as np
def inverse_permutation(indices):
 a=np.asarray(indices)
 if a.ndim!=1 or (a.size and not np.issubdtype(a.dtype,np.integer)): raise ValueError()
 if np.any(a<0) or np.any(a>=a.size): raise ValueError()
 out=np.empty(a.size,dtype=np.int64);out[a.astype(np.intp)]=np.arange(a.size);return out
