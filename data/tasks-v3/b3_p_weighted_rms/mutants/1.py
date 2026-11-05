import numpy as np
def weighted_rms(values,weights):
 x=np.asarray(values,dtype=float); w=np.asarray(weights,dtype=float)
 return float(np.sqrt(np.mean(w*x*x)))
