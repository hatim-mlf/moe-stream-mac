import numpy as np
def weighted_rms(values,weights):
 x=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
 if x.ndim!=1 or w.shape!=x.shape or np.any(w<0) or w.sum()<=0: raise ValueError()
 return float(np.sqrt(np.sum(w*x*x)/w.sum()))
