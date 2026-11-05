import numpy as np

def linear_resample_1d(query, xp, fp):
    q=np.asarray(query,dtype=np.float64); x=np.asarray(xp,dtype=np.float64); y=np.asarray(fp,dtype=np.float64)
    if x.ndim!=1 or y.ndim!=1 or x.size<2 or x.size!=y.size or not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(q).all() or np.any(np.diff(x)<=0):
        raise ValueError("invalid samples")
    return np.interp(q,x,y).reshape(q.shape)
