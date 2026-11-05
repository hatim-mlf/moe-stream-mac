import numpy as np

def weighted_blend32(a, b, weight):
    aa=np.asarray(a); bb=np.asarray(b); ww=np.asarray(weight)
    if any(v.dtype.kind not in "iuf" for v in (aa,bb,ww)):
        raise TypeError("inputs must be real numeric arrays")
    if not np.all(np.isfinite(ww)) or np.any(ww<0) or np.any(ww>1):
        raise ValueError("weight must be finite and in [0, 1]")
    try:
        return aa.astype(np.float64)*ww + bb.astype(np.float64)*(1-ww)
    except ValueError as exc:
        raise ValueError("inputs are not broadcastable") from exc
