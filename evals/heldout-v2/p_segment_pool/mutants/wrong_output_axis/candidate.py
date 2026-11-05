import numpy as np

def segment_reduce_sum(values, segment_ids, n_segments):
    v=np.asarray(values); ids=np.asarray(segment_ids)
    if v.ndim<1 or ids.ndim!=1 or ids.shape[0]!=v.shape[0]:
        raise ValueError("segment ids must match the leading dimension")
    if v.dtype.kind not in "iuf" or ids.dtype.kind not in "iu":
        raise TypeError("values must be real numeric and ids integer")
    if isinstance(n_segments,(bool,np.bool_)) or not isinstance(n_segments,(int,np.integer)):
        raise TypeError("n_segments must be an integer")
    n=int(n_segments)
    if n<0 or np.any(ids<0) or np.any(ids>=n):
        raise ValueError("segment id out of range")
    out=np.zeros((n,)+v.shape[1:],dtype=np.float64)
    if v.ndim>1: out=np.zeros(v.shape[1:]+(n,),dtype=np.float64)
    for i,seg in enumerate(ids): out[int(seg)] += v[i].astype(np.float64,copy=False)
    return out
