import numpy as np

def window_matrix_copy(values, width, step):
    a=np.asarray(values)
    if a.ndim<1: raise ValueError("values must have a leading axis")
    if isinstance(width,(bool,np.bool_)) or not isinstance(width,(int,np.integer)) or isinstance(step,(bool,np.bool_)) or not isinstance(step,(int,np.integer)):
        raise TypeError("width and step must be integers")
    w,st=int(width),int(step)
    if w<=0 or st<=0 or w>a.shape[0]: raise ValueError("invalid window parameters")
    starts=range(0,a.shape[0]-w+1,st)
    out=np.empty((len(starts),w)+a.shape[1:],dtype=a.dtype)
    for j,start in enumerate(starts): out[j]=a[start:start+w:st]
    return out
