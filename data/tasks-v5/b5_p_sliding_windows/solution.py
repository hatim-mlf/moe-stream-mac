import numpy as np

def sliding_windows_1d(values, width, step=1):
    a=np.asarray(values)
    valid=lambda z:isinstance(z,(int,np.integer)) and not isinstance(z,(bool,np.bool_))
    if a.ndim!=1 or not valid(width) or not valid(step) or width<=0 or step<=0: raise ValueError
    count=max(0,1+(len(a)-int(width))//int(step))
    out=np.empty((count,int(width)),dtype=a.dtype)
    for i in range(count): out[i]=a[i*int(step):i*int(step)+int(width)]
    return out
