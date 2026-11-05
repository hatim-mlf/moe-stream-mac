import numpy as np
def block_max_1d(values,width):
    a=np.asarray(values)
    return np.asarray([np.max(a[i:i+width]) for i in range(0,len(a)-width+1,width)])
