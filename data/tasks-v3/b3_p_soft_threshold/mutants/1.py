import numpy as np
def soft_threshold_array(x,threshold):
 a=np.asarray(x,dtype=float); return np.sign(a)*np.maximum(np.abs(a)-threshold,0)
