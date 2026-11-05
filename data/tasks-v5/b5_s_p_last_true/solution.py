import numpy as np

def last_true_index(mask):
    a=np.asarray(mask)
    if a.ndim!=1 or a.dtype!=np.bool_: raise ValueError
    ids=np.flatnonzero(a)
    return int(ids[-1]) if len(ids) else -1
