Target: candidate.py (Python 3.14; standard library and NumPy only)

Dequantize low-first unsigned 4-bit values with one affine scale and bias per fixed group.

Exact public signature:
```python
def dequant_byte_groups(packed, scales, biases, group_size): ...
```

`packed` must be a rank-1 uint8 array; each byte stores even value in low nibble, odd value in high nibble. Nibbles are unsigned 0..15. Scales and biases must be equal-length rank-1 real numeric arrays. `group_size` must be a positive integer (not bool); unpacked length must divide evenly by it, and scale/bias count must equal length/group_size. Raise TypeError for dtype/type violations and ValueError for shape/layout violations. Return float32 values `q[i]*scale[i//group_size] + bias[...]`.

Return only the complete file in one fenced python block.

## Your previous attempt

```python
import numpy as np

def dequant_byte_groups(packed, scales, biases, group_size):
    p=np.asarray(packed); s=np.asarray(scales); b=np.asarray(biases)
    if p.ndim!=1 or p.dtype!=np.uint8: raise TypeError("packed must be a uint8 vector")
    if s.ndim!=1 or b.ndim!=1 or s.shape!=b.shape: raise ValueError("scale and bias vectors must match")
    if s.dtype.kind not in "iuf" or b.dtype.kind not in "iuf": raise TypeError("scales and biases must be real")
    if isinstance(group_size,bool) or not isinstance(group_size,(int,np.integer)): raise TypeError("group_size must be integer")
    g=int(group_size); n=p.size*2
    if g<=0 or n%g or s.size!=n//g: raise ValueError("group layout mismatch")
    q=np.empty(n,dtype=np.uint8); q[0::2]=p&15; hi=(p>>4).astype(np.int16); q[1::2]=np.where(hi>=8,hi-16,hi).astype(np.uint8)
    ids=np.arange(n)//g
    return (q.astype(np.float32)*s.astype(np.float32)[ids]+b.astype(np.float32)[ids]).astype(np.float32)
```

## Review notes

Treat every nibble as unsigned and use integer division by group_size with no offset.
