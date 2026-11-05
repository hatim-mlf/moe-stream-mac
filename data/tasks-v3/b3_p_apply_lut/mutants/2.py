import numpy as np
def apply_integer_lut(indices,lut):
 i=np.asarray(indices); t=np.asarray(lut)
 if i.dtype.kind not in "iu" or t.ndim!=1 or t.size==0 or np.any(i<0) or np.any(i>=t.size): raise ValueError()
 return t[i].astype(np.int64)
