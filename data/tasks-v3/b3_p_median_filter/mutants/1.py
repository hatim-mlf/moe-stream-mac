import numpy as np
def median_filter_valid(x,width):
 a=np.asarray(x,dtype=float)
 return np.array([np.median(a[i:i+width]) for i in range(a.size-width+1)])
