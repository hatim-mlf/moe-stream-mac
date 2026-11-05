import numpy as np

def pack_i16le(values):
    a=np.asarray(values)
    if a.ndim!=1 or a.dtype.kind not in "iu" or (a.size and (np.any(a < -32768) or np.any(a > 32767))): raise ValueError("invalid int16 values")
    return a.astype("<i2").tobytes()
