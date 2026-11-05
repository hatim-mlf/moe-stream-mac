import numpy as np
def masked_softmax_rows(logits,mask):
    x=np.asarray(logits,dtype=np.float64)
    m=np.asarray(mask,dtype=bool)
    out=np.zeros_like(x)
    for i in range(x.shape[0]):
        e=np.exp(x[i]-np.max(x[i]))
        out[i]=e/e.sum()
    return out
