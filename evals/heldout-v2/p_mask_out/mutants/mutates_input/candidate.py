import numpy as np

def masked_scale_copy(values, mask, factor):
    a=np.asarray(values); m=np.asarray(mask)
    if a.dtype.kind!='f': raise TypeError("values must have floating dtype")
    if m.dtype.kind!='b': raise TypeError("mask must be boolean")
    if not np.isscalar(factor) or np.asarray(factor).dtype.kind not in "iuf" or not np.isfinite(factor): raise ValueError("factor must be a finite real scalar")
    try: mb=np.broadcast_to(m,a.shape)
    except ValueError as exc: raise ValueError("mask is not broadcastable") from exc
    f=np.asarray(factor,dtype=a.dtype)
    a[mb] *= f
    return a
