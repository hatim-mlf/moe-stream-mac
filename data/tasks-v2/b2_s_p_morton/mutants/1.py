import numpy as np
def decode_morton2d(codes,bits=16):
 a=np.asarray(codes)
 if a.ndim!=1 or not np.issubdtype(a.dtype,np.integer) or not isinstance(bits,(int,np.integer)) or isinstance(bits,(bool,np.bool_)) or not 1<=bits<=16 or np.any(a<0) or np.any(a>=(1<<(2*bits))): raise ValueError()
 out=np.zeros((a.size,2),dtype=np.int64)
 for bit in range(bits):
  out[:,0]|=((a>>(2*bit+1))&1).astype(np.int64)<<bit
  out[:,1]|=((a>>(2*bit))&1).astype(np.int64)<<bit
 return out
