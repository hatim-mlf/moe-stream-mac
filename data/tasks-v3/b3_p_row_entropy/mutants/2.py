import numpy as np
def row_entropy(probabilities):
 p=np.asarray(probabilities,dtype=float)
 if p.ndim!=2 or not np.isfinite(p).all() or np.any(p<0) or np.any(p.sum(1)<=0): raise ValueError()
 p=p/p.sum(1,keepdims=True)
 t=np.zeros_like(p);np.log(p,out=t,where=p>0)
 return -np.sum(np.where(p>0,p*t/np.log(2),0),axis=1)
