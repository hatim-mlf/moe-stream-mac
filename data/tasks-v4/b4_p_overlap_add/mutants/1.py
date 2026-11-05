import numpy as np
def overlap_add_frames(frames,hop,length):
    a=np.asarray(frames,dtype=float)
    if a.ndim!=2 or hop<=0 or length<0: raise ValueError()
    out=np.zeros(length)
    for i,row in enumerate(a):
        start=i*hop; stop=min(length,start+len(row))
        if start<length: out[start:stop]+=row[:stop-start]
    return out
