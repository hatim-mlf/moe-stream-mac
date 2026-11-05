import math
import numpy as np

def cumulative_logsumexp(values):
    a=np.asarray(values,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all(): raise ValueError
    out=np.empty(len(a),dtype=np.float64); m=-math.inf; s=0.0
    for i,v in enumerate(a):
        if v>m: s=s*math.exp(m-v)+1.0; m=float(v)
        else: s+=math.exp(float(v)-m)
        out[i]=m+math.log(s)
    return out
