def decode_uleb32(data,offset=0):
    value=0; shift=0
    for byte in data[offset:]:
        value |= (byte&127)<<shift
        if byte<128: return value,offset+shift//7+1
        shift += 7
    raise ValueError("truncated")
