import numpy as np
def polyline_length2d(points):
 p=np.asarray(points,dtype=float)
 return float(np.sqrt(np.sum(np.diff(p,axis=0)**2)))
