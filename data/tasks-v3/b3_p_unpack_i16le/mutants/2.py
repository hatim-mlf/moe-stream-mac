import numpy as np
def unpack_i16le(data):
 if not isinstance(data,(bytes,bytearray,memoryview)) or len(data)%2: raise ValueError()
 return np.frombuffer(data,dtype=">i2").astype(np.int16,copy=True)
