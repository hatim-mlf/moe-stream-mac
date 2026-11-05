import numpy as np
def finite_counts_2d(values):
    return np.isfinite(np.asarray(values)).sum(axis=0)
