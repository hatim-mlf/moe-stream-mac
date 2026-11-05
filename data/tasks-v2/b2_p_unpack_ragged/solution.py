import numpy as np
def unpack_ragged(flat, offsets):
    a = np.asarray(flat)
    o = np.asarray(offsets)
    if a.ndim != 1 or o.ndim != 1 or o.size == 0 or not np.issubdtype(o.dtype, np.integer):
        raise ValueError("invalid ranks or offsets dtype")
    if o[0] != 0 or o[-1] != a.size or np.any(o[1:] < o[:-1]):
        raise ValueError("invalid offset sequence")
    return [a[int(o[i]):int(o[i + 1])].copy() for i in range(o.size - 1)]
