import numpy as np
def sliding_dot_valid(signal,kernel):
 x=np.asarray(signal,dtype=np.float64); k=np.asarray(kernel,dtype=np.float64)
 if x.ndim!=1 or k.ndim!=1 or not k.size or not np.isfinite(x).all() or not np.isfinite(k).all(): raise ValueError()
 if k.size>x.size:return np.empty(0,dtype=np.float64)
 return np.array([np.dot(x[i:i+k.size],k[::-1]) for i in range(x.size-k.size+1)])
