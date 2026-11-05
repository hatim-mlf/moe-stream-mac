import numpy as np
def one_hot_matrix(indices,classes):
    a=np.asarray(indices)
    if a.ndim!=1 or (a.size and a.dtype.kind not in "iu") or not isinstance(classes,(int,np.integer)) or classes<=0:
        raise ValueError("invalid indices or class count")
    if np.any(a<0) or np.any(a>=classes):
        raise ValueError("index out of range")
    out=np.zeros((a.size,int(classes)),dtype=np.uint8)
    out[np.arange(a.size),a.astype(np.intp,copy=False)]=1
    return out
