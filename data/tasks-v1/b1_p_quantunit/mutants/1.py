import numpy as np
def quantize_unit_u8(values,low,high):
    x=np.asarray(values,dtype=float)
    scaled=(np.clip(x,low,high)-low)*255/(high-low)
    return np.clip(scaled.astype(np.int64),0,255).astype(np.uint8)
