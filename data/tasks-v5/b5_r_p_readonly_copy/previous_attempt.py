import numpy as np

def readonly_c_copy(array):
    out=np.ascontiguousarray(array);out.setflags(write=False);return out
