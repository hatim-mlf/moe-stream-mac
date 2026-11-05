import numpy as np
def column_zscore(x):
 a=np.asarray(x,dtype=np.float64)
 if a.ndim!=2 or a.shape[0]==0 or not np.isfinite(a).all(): raise ValueError()
 m=a.mean(axis=0); s=a.std(axis=0,ddof=1); out=np.zeros_like(a); np.divide(a-m,s,out=out,where=s!=0); return out
