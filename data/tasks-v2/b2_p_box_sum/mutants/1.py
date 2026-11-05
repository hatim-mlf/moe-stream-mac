import numpy as np
def box_sum_valid(x,height,width):
 a=np.asarray(x,dtype=np.float64)
 if a.ndim!=2 or not np.isfinite(a).all() or not isinstance(height,(int,np.integer)) or isinstance(height,(bool,np.bool_)) or not isinstance(width,(int,np.integer)) or isinstance(width,(bool,np.bool_)) or height<=0 or width<=0: raise ValueError()
 if height>a.shape[0] or width>a.shape[1]: return np.empty((max(0,a.shape[0]-height+1),max(0,a.shape[1]-width+1)))
 p=np.pad(a,((1,0),(1,0))).cumsum(0).cumsum(1)
 h,w=int(height),int(width)
 return p[h:,w:]-p[:-h,w:]-p[h:,:-w]
