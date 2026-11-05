import numpy as np
def dilate_mask_1d(mask,radius):
 a=np.asarray(mask)
 if a.ndim!=1 or not isinstance(radius,(int,np.integer)) or isinstance(radius,(bool,np.bool_)) or radius<0: raise ValueError()
 s=a.astype(bool); out=np.zeros(s.size,dtype=bool)
 for i in range(s.size): out[i]=np.any(s[max(0,i-radius):min(s.size,i+radius)])
 return out
