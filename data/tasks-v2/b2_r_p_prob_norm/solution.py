import numpy as np
def normalize_prob_vector(weights):
 a=np.asarray(weights,dtype=np.float64)
 if a.ndim!=1 or not np.isfinite(a).all() or np.any(a<0) or a.sum()<=0: raise ValueError("invalid probability weights")
 return a/a.sum()
