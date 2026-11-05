import numpy as np
def true_spans(mask):
    a=np.asarray(mask)
    if a.ndim!=1 or a.dtype!=np.bool_:
        raise ValueError("expected a boolean vector")
    if not a.size:
        return []
    starts=np.flatnonzero(a & np.r_[True,~a[:-1]])
    stops=np.flatnonzero(a & np.r_[~a[1:],True])+1
    return list(zip(starts.tolist(),stops.tolist()))
