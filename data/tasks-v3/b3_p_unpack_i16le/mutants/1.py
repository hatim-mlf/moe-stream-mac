import numpy as np
def unpack_i16le(data): return np.frombuffer(data,dtype="<i2")
