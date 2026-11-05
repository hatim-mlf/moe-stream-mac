import numpy as np
def expand_counts(values,counts):
    v=np.asarray(values)
    c=np.asarray(counts)
    if v.ndim!=1 or c.ndim!=1 or len(v)!=len(c) or (c.size and c.dtype.kind not in "iu") or np.any(c<0):
        raise ValueError("values and nonnegative integer counts must be matching vectors")
    return np.repeat(v,c.astype(np.intp,copy=False))
