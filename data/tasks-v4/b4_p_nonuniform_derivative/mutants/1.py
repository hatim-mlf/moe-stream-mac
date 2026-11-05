import numpy as np
def derivative_nonuniform(y,x):
    v=np.asarray(y,dtype=float); t=np.asarray(x,dtype=float)
    if v.ndim!=1 or t.ndim!=1 or len(v)!=len(t) or len(v)<2 or np.any(np.diff(t)<=0): raise ValueError()
    s=np.diff(v)/np.diff(t); out=np.empty_like(v); out[0]=s[0]; out[-1]=s[-1]
    if len(v)>2: out[1:-1]=(s[:-1]+s[1:])/2
    return out
