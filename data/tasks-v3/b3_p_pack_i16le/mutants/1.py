import numpy as np
def pack_i16le(values): return np.asarray(values,dtype="<i2").tobytes()
