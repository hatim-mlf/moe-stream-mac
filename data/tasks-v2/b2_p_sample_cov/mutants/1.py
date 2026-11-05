import numpy as np
def sample_covariance(x):
 a=np.asarray(x,dtype=np.float64)
 if a.ndim!=2 or a.shape[0]<2 or not np.isfinite(a).all(): raise ValueError()
 c=a-a.mean(axis=0); return (c.T@c)/a.shape[0]
