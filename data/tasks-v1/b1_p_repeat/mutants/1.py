import numpy as np
def expand_counts(values,counts):
    return np.repeat(np.asarray(values),np.asarray(counts,dtype=int))
