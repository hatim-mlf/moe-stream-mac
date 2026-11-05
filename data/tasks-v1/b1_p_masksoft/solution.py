import numpy as np
def masked_softmax_rows(logits,mask):
    x=np.asarray(logits,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    if x.ndim!=2 or m.shape!=x.shape:
        raise ValueError("expected matching matrices")
    out=np.zeros(x.shape,dtype=np.float64)
    for i in range(x.shape[0]):
        active=m[i]
        if np.any(active):
            vals=x[i,active]
            exp=np.exp(vals-np.max(vals))
            out[i,active]=exp/np.sum(exp)
    return out
