import numpy as np
def unpack_ragged(flat,offsets):
 a=np.asarray(flat); o=np.asarray(offsets)
 if a.ndim!=1 or o.ndim!=1 or not o.size or not np.issubdtype(o.dtype,np.integer) or o[0]!=0 or o[-1]!=a.size or np.any(o[1:]<o[:-1]): raise ValueError()
 return [a[int(o[i]):int(o[i+1])].copy() for i in range(1,o.size)]
