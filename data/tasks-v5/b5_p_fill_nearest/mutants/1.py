import numpy as np

def fill_nearest_1d(values):
    a=np.asarray(values,dtype=float); out=a.copy()
    for i in range(len(out)):
        if np.isnan(out[i]):
            j=max(k for k in range(i) if np.isfinite(out[k])) if any(np.isfinite(out[:i])) else next(k for k in range(i+1,len(out)) if np.isfinite(out[k]))
            out[i]=out[j]
    return out
