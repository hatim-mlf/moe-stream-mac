import numpy as np

def indices_above_threshold(scores,threshold):
    a=np.asarray(scores,dtype=float);return np.flatnonzero(a>=threshold).astype(np.int64)
