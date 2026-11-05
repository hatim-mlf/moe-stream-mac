import numpy as np
def project_simplex_rows(x):
    a=np.asarray(x,dtype=float)
    if a.ndim!=2 or a.shape[1]==0: raise ValueError()
    out=np.maximum(a,0)
    return out / np.maximum(out.sum(axis=1,keepdims=True),1)
