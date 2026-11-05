import numpy as np
def ema_series(values,alpha):
    x=np.asarray(values,dtype=float)
    out=np.zeros_like(x)
    for i in range(len(x)): out[i]=alpha*x[i]+(1-alpha)*(out[i-1] if i else 0)
    return out
