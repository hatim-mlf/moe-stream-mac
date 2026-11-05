def decode_records_u32be(data):
    if not isinstance(data,(bytes,bytearray,memoryview)): raise ValueError("expected bytes-like input")
    raw=bytes(data); pos=0; out=[]
    while pos<len(raw):
        if len(raw)-pos<4: raise ValueError("truncated header")
        size=int.from_bytes(raw[pos:pos+4],"big"); pos+=4
        if size>len(raw)-pos: raise ValueError("truncated payload")
        out.append(raw[pos:pos+size]); pos+=size
    return out
