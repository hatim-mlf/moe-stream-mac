import numpy as np
def row_entropy(probabilities):
 p=np.asarray(probabilities,dtype=float)
 with np.errstate(divide="ignore",invalid="ignore"): return -np.sum(p*np.log(p),axis=1)
