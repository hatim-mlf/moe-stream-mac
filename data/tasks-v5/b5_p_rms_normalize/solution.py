import numpy as np

def normalize_by_rms(x,axis=-1,eps=0.0):
    a=np.asarray(x,dtype=np.float64)
    valid=isinstance(axis,(int,np.integer)) and not isinstance(axis,(bool,np.bool_))
    if a.ndim<1 or not valid or not np.isfinite(a).all() or not np.isscalar(eps) or isinstance(eps,(bool,np.bool_)) or not np.isfinite(eps) or eps<0: raise ValueError
    ax=int(axis); ax=ax+a.ndim if ax<0 else ax
    if ax<0 or ax>=a.ndim: raise ValueError
    d=np.sqrt(np.mean(a*a,axis=ax,keepdims=True)+float(eps))
    return np.divide(a,d,out=np.zeros_like(a),where=d!=0)
