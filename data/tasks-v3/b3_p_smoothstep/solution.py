import numpy as np

def smoothstep01(x):
    a=np.asarray(x,dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError("nonfinite input")
    t=np.clip(a,0.0,1.0)
    return t*t*(3.0-2.0*t)
