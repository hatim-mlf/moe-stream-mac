import numpy as np
def quantize_symmetric_codes(values,scale):
    return np.clip(np.rint(np.asarray(values,dtype=float)/scale),-128,127).astype(np.int8)
