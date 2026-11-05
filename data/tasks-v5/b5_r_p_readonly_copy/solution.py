import numpy as np

def readonly_c_copy(array):
    out=np.array(array,copy=True,order='C',subok=False)
    out.setflags(write=False)
    return out
