import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
def median_filter_valid(x,width):
 a=np.asarray(x,dtype=float)
 if a.ndim!=1 or width<1: raise ValueError()
 if width>a.size:return np.empty(0)
 return np.median(sliding_window_view(a,width),axis=0)
