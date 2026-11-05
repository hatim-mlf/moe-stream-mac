import numpy as np

def last_true_index(mask):
    a=np.asarray(mask)
    ids=np.flatnonzero(a)
    return int(ids[0]) if len(ids) else -1
