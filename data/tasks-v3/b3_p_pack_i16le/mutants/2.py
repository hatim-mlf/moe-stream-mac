import numpy as np
def pack_i16le(values):
 a=np.asarray(values)
 if a.ndim!=1 or a.dtype.kind not in "iu": raise ValueError()
 return a.astype(">i2").tobytes()
