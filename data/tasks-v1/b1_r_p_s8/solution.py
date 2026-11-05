import numpy as np
def quantize_symmetric_codes(values,scale):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(scale) or scale<=0 or not np.all(np.isfinite(x)):
        raise ValueError("invalid scale or values")
    return np.clip(np.rint(x/scale),-127,127).astype(np.int8)
