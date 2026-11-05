def decode_records_u32be(data):
 raw=bytes(data); p=0; out=[]
 while p<len(raw):
  if len(raw)-p<4: raise ValueError()
  n=int.from_bytes(raw[p:p+4],"little"); p+=4
  if n>len(raw)-p: raise ValueError()
  out.append(raw[p:p+n]); p+=n
 return out
