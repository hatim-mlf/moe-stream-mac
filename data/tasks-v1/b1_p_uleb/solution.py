def decode_uleb32(data,offset=0):
    view=memoryview(data).cast("B")
    if not isinstance(offset,int) or offset<0 or offset>len(view):
        raise ValueError("invalid offset")
    value=0
    for index in range(5):
        pos=offset+index
        if pos>=len(view):
            raise ValueError("truncated value")
        byte=int(view[pos])
        if index==4 and byte>0x0f:
            raise ValueError("uint32 overflow")
        value|=(byte&0x7f)<<(7*index)
        if byte<0x80:
            return value,pos+1
    raise ValueError("value exceeds five bytes")
