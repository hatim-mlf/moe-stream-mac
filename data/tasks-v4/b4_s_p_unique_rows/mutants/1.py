import numpy as np
def first_unique_rows(x):
    a=np.asarray(x)
    if a.ndim!=2: raise ValueError()
    return np.unique(a,axis=0,return_index=True)[1].astype(np.int64)
