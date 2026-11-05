import numpy as np

def diagonal_means(matrix):
    # TODO_BEGIN
    a=np.asarray(matrix)
    if a.ndim!=2 or a.shape[0]==0 or a.shape[1]==0:
        raise ValueError("matrix must be nonempty and two-dimensional")
    if a.dtype.kind not in "iuf" or not np.all(np.isfinite(a)):
        raise ValueError("matrix must contain finite real values")
    result=[]
    for offset in range(-(a.shape[0]-1),a.shape[1]):
        diag=np.diagonal(a,offset=offset)
        result.append(np.mean(diag,dtype=np.float64))
    return np.asarray(result,dtype=np.float64)
    # TODO_END
