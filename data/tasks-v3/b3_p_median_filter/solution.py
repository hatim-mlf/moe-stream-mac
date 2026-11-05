import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def median_filter_valid(x, width):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all() or isinstance(width,bool) or not isinstance(width,(int,np.integer)) or width<=0: raise ValueError("invalid window")
    if width>a.size: return np.empty(0,dtype=np.float64)
    return np.median(sliding_window_view(a,width),axis=1)
