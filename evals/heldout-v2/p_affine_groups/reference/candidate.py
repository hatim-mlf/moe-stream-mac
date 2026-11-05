import numpy as np

def dequant_byte_groups(packed, scales, biases, group_size):
    p=np.asarray(packed); s=np.asarray(scales); b=np.asarray(biases)
    if p.ndim!=1 or p.dtype!=np.uint8: raise TypeError("packed must be a uint8 vector")
    if s.ndim!=1 or b.ndim!=1 or s.shape!=b.shape: raise ValueError("scale and bias vectors must match")
    if s.dtype.kind not in "iuf" or b.dtype.kind not in "iuf": raise TypeError("scales and biases must be real")
    if isinstance(group_size,bool) or not isinstance(group_size,(int,np.integer)): raise TypeError("group_size must be integer")
    g=int(group_size); n=p.size*2
    if g<=0 or n%g or s.size!=n//g: raise ValueError("group layout mismatch")
    q=np.empty(n,dtype=np.uint8); q[0::2]=p&15; q[1::2]=p>>4
    ids=np.arange(n)//g
    return (q.astype(np.float32)*s.astype(np.float32)[ids]+b.astype(np.float32)[ids]).astype(np.float32)
