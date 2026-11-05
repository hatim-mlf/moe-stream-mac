import numpy as np
def means_by_label(values,labels):
 x=np.asarray(values,dtype=np.float64); ids=np.asarray(labels)
 if x.ndim!=1 or ids.ndim!=1 or x.size==0 or x.size!=ids.size or not np.issubdtype(ids.dtype,np.integer) or not np.isfinite(x).all(): raise ValueError("invalid values or labels")
 keys=sorted(set(int(v) for v in ids))
 means=np.array([x[ids==key].mean() for key in keys],dtype=np.float64)
 return np.asarray(keys,dtype=np.int64),means
