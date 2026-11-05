import numpy as np
def pack_ragged(rows):
    arrays = [np.asarray(row) for row in rows]
    if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.number) for a in arrays):
        raise ValueError("rows must be one-dimensional numeric arrays")
    dtype = np.result_type(*(a.dtype for a in arrays)) if arrays else np.float64
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    for i, a in enumerate(arrays): offsets[i + 1] = offsets[i] + a.size
    flat = np.concatenate([a.astype(dtype, copy=False) for a in arrays]) if arrays else np.empty(0, dtype=dtype)
    return flat, offsets
