import numpy as np
def pad_numeric_rows(rows,pad_value=0):
    arrays=[np.asarray(row) for row in rows]
    if any(a.ndim!=1 or a.dtype.kind not in "iuf" for a in arrays):
        raise ValueError("rows must be one-dimensional numeric arrays")
    if arrays:
        dtype=np.result_type(*(a.dtype for a in arrays),np.asarray(pad_value).dtype)
    else:
        dtype=np.float64
    lengths=np.asarray([a.size for a in arrays],dtype=np.int64)
    width=int(lengths.max()) if lengths.size else 0
    matrix=np.full((len(arrays),width),pad_value,dtype=dtype)
    for i,a in enumerate(arrays):
        matrix[i,:a.size]=a
    return matrix,lengths
