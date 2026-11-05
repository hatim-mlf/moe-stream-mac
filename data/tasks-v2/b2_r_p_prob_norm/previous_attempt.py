import numpy as np
def normalize_prob_vector(weights):
 a=np.asarray(weights,dtype=np.float64)
 if a.ndim!=1 or a.size==0 or not np.isfinite(a).all(): raise ValueError("invalid probability vector")
 return a/a.sum()
