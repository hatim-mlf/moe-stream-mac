import numpy as np
def spectral_norm_2x2(x):
    a=np.asarray(x,dtype=float)
    if a.ndim<2 or a.shape[-2:]!=(2,2): raise ValueError()
    return np.sqrt(np.sum(a*a,axis=(-2,-1)))
