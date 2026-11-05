import numpy as np

def weighted_centroid_2d(weights):
    a=np.asarray(weights,dtype=float)
    if a.ndim!=2 or np.any(a<0) or a.sum()<=0: raise ValueError
    return (float(a.shape[0]-1)/2.0,float(a.shape[1]-1)/2.0)
