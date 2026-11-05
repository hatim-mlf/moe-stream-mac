import numpy as np
def linear_resample_1d(query,xp,fp):
 q=np.asarray(query,dtype=float); x=np.asarray(xp,dtype=float); y=np.asarray(fp,dtype=float)
 if x.ndim!=1 or y.ndim!=1 or x.size!=y.size or x.size<2 or np.any(np.diff(x)<=0): raise ValueError()
 return np.interp(q,x,y).ravel()
