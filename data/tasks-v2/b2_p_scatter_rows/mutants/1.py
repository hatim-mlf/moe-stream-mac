import numpy as np
def scatter_add_rows(base,indices,updates):
 a=np.asarray(base,dtype=np.float64); i=np.asarray(indices); u=np.asarray(updates,dtype=np.float64)
 if a.ndim!=2 or i.ndim!=1 or not np.issubdtype(i.dtype,np.integer) or u.shape!=(len(i),a.shape[1]): raise ValueError()
 if np.any(i<0) or np.any(i>=a.shape[0]): raise ValueError()
 out=a.copy(); out[i]+=u; return out
