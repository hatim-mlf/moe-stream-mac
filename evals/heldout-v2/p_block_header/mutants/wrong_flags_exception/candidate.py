import struct

def unpack_frame_header(data):
    if not isinstance(data,(bytes,bytearray,memoryview)):
        raise TypeError("data must be bytes-like")
    blob=bytes(data)
    if len(blob)<12: raise ValueError("truncated frame")
    magic,version,flags,length,sequence=struct.unpack_from("<4sBBHI",blob,0)
    if magic!=b"RBUF" or version!=2 or flags & ~7:
        raise TypeError("invalid frame header")
    if len(blob)!=12+length:
        raise ValueError("payload length mismatch")
    return flags,length,sequence
