import numpy as np
def quantize_unit_u8(values,low,high):
    x=np.asarray(values,dtype=np.float64)
    if not np.isfinite(low) or not np.isfinite(high) or high<=low or not np.all(np.isfinite(x)):
        raise ValueError("invalid quantization range or values")
    scaled=(np.clip(x,low,high)-low)*(255.0/(high-low))
    return np.floor(scaled+0.5).astype(np.uint8)
