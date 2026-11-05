import numpy as np

def weighted_centroid_2d(weights):
    a=np.asarray(weights,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all() or np.any(a<0) or a.sum()<=0: raise ValueError
    scaled=a/float(a.max()); total=float(scaled.sum())
    r=float(np.dot(np.arange(a.shape[0]),scaled.sum(axis=1))/total)
    c=float(np.dot(np.arange(a.shape[1]),scaled.sum(axis=0))/total)
    return r,c
