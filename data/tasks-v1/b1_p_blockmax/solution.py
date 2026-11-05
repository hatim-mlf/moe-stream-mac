import numpy as np
def block_max_1d(values,width):
    a=np.asarray(values)
    if a.ndim!=1 or not isinstance(width,(int,np.integer)) or width<=0:
        raise ValueError("expected a vector and positive width")
    if not a.size:
        return a.copy()
    return np.asarray([np.max(a[i:i+width]) for i in range(0,a.size,width)],dtype=a.dtype)
