import numpy as np

def apply_integer_lut(indices, lut):
    i=np.asarray(indices); table=np.asarray(lut)
    if i.dtype.kind not in "iu" or table.ndim!=1 or table.size==0 or (i.size and (np.any(i<0) or np.any(i>=table.size))): raise ValueError("invalid lookup")
    return table[i]
