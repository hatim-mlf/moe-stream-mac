import math
import numpy as np

def clip_global_norm(tensors,max_norm):
    if isinstance(max_norm,(bool,np.bool_)) or not np.isscalar(max_norm) or not np.isfinite(max_norm) or max_norm<0: raise ValueError
    arrays=[np.asarray(t,dtype=np.float64) for t in tensors]
    if any(not np.isfinite(a).all() for a in arrays): raise ValueError
    norm=math.sqrt(sum(float(np.sum(a*a)) for a in arrays))
    factor=min(1.0,float(max_norm)/norm) if norm else 1.0
    return [a.copy()*factor for a in arrays],norm
