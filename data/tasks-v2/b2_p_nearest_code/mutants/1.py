import numpy as np
def nearest_codebook(values,centers):
 x=np.asarray(values,dtype=np.float64); c=np.asarray(centers,dtype=np.float64)
 if x.ndim!=1 or c.ndim!=1 or not c.size or not np.isfinite(x).all() or not np.isfinite(c).all(): raise ValueError()
 d=np.abs(x[:,None]-c[None,:]); return (c.size-1-np.argmin(d[:,::-1],axis=1)).astype(np.int64)
