import numpy as np
def weighted_rows_mean(x,weights):
 a=np.asarray(x,dtype=np.float64); w=np.asarray(weights,dtype=np.float64)
 if a.ndim!=2 or w.ndim!=1 or w.size!=a.shape[0] or not np.isfinite(a).all() or not np.isfinite(w).all() or np.any(w<0) or w.sum()<=0: raise ValueError()
 return np.sum(a*w[:,None],axis=1)/w.sum()
