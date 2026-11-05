import numpy as np
def owned_float32_c_array(values):
    return np.array(values,dtype=np.float32,copy=True,order="C")
