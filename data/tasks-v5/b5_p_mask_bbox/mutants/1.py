import numpy as np

def bbox_from_mask(mask):
    a=np.asarray(mask,dtype=bool)
    if a.ndim!=2:raise ValueError
    r,c=np.nonzero(a)
    if not len(r):return None
    return int(r.min()),int(c.min()),int(r.max()),int(c.max())
