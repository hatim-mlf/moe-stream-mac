import numpy as np

def stable_rank_rows(scores, group_ids, limit):
    s=np.asarray(scores); g=np.asarray(group_ids)
    if s.ndim!=1 or g.ndim!=1 or s.shape!=g.shape:
        raise ValueError("scores and groups must be equal-length vectors")
    if s.dtype.kind not in "iuf" or g.dtype.kind not in "iu":
        raise TypeError("scores must be real and group ids integer")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if isinstance(limit,(bool,np.bool_)) or not isinstance(limit,(int,np.integer)):
        raise TypeError("limit must be an integer")
    k=int(limit)
    if k<0 or k>s.size: raise ValueError("limit out of range")
    idx=np.arange(s.size,dtype=np.int64)
    return np.lexsort((idx,-s,g))[:k]
