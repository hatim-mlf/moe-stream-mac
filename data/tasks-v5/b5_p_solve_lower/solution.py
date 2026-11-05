import numpy as np

def solve_lower_triangular(matrix,rhs):
    a=np.asarray(matrix,dtype=np.float64); b=np.asarray(rhs,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or b.ndim!=1 or len(b)!=len(a) or not np.isfinite(a).all() or not np.isfinite(b).all(): raise ValueError
    n=len(b)
    if np.any(np.triu(a,1)!=0) or np.any(np.diag(a)==0): raise ValueError
    out=np.empty(n,dtype=np.float64)
    for i in range(n): out[i]=(b[i]-np.dot(a[i,:i],out[:i]))/a[i,i]
    return out
