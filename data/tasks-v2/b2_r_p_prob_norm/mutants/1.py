import numpy as np
def normalize_prob_vector(weights):
 a=np.asarray(weights,dtype=np.float64)
 return a/a.sum()
