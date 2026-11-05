import numpy as np
def ema_series(values,alpha):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or not np.all(np.isfinite(x)) or not np.isfinite(alpha) or not 0<alpha<=1:
        raise ValueError("invalid values or alpha")
    out=np.empty(x.shape,dtype=np.float64)
    if x.size:
        out[0]=x[0]
        for i in range(1,x.size):
            out[i]=alpha*x[i]+(1-alpha)*out[i-1]
    return out
