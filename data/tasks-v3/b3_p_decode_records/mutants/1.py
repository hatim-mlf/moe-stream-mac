def decode_records_u32be(data):
 raw=bytes(data); p=0; out=[]
 while p+4<=len(raw):
  n=int.from_bytes(raw[p:p+4],"big"); p+=4; out.append(raw[p:p+n]); p+=n
 return out
