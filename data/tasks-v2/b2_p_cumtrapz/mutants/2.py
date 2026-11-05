import numpy as np
def cumulative_trapezoid(y,x):
 a=np.asarray(y,dtype=np.float64); t=np.asarray(x,dtype=np.float64)
 if a.ndim!=1 or t.ndim!=1 or a.size!=t.size or not np.isfinite(a).all() or not np.isfinite(t).all() or np.any(t[1:]<=t[:-1]): raise ValueError()
 out=np.zeros(a.size,dtype=np.float64)
 if a.size>1: out[1:]=np.cumsum((t[1:]-t[:-1])*(a[1:]+a[:-1])*0.5)
 if out.size: out[0]=a[0]
 return out
