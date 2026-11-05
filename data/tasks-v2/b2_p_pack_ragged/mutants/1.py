import numpy as np
def pack_ragged(rows):
 a=[np.asarray(r) for r in rows]
 if any(x.ndim!=1 or not np.issubdtype(x.dtype,np.number) for x in a): raise ValueError()
 dt=np.result_type(*(x.dtype for x in a)) if a else np.float64; off=np.zeros(len(a)+1,dtype=np.int64)
 for i,x in enumerate(a): off[i+1]=off[i]+x.size
 flat=np.concatenate([x.astype(dt,copy=False) for x in a]) if a else np.empty(0,dtype=dt)
 return flat[::-1],off
