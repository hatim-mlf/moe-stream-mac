import numpy as np
def soft_threshold_array(x,threshold):
 a=np.asarray(x,dtype=float)
 if threshold<0: raise ValueError()
 return np.sign(a)*np.maximum(np.abs(a)-threshold,0)
