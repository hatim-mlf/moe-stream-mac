import numpy as np

def unpack_i16le(data):
    if not isinstance(data,(bytes,bytearray,memoryview)): raise ValueError("expected bytes-like data")
    raw=bytes(data)
    if len(raw)%2: raise ValueError("odd byte count")
    return np.frombuffer(raw,dtype="<i2").astype(np.int16,copy=True)
