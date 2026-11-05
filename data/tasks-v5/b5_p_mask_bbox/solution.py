import numpy as np

def bbox_from_mask(mask):
    a=np.asarray(mask)
    if a.ndim!=2 or a.dtype!=np.bool_: raise ValueError
    r,c=np.nonzero(a)
    if not len(r):return None
    return int(r.min()),int(c.min()),int(r.max()+1),int(c.max()+1)
