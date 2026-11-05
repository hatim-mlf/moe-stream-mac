import numpy as np

def solve_lower_triangular(matrix,rhs):
    a=np.asarray(matrix,dtype=float); b=np.asarray(rhs,dtype=float); n=len(b); x=np.empty(n)
    for i in range(n-1,-1,-1): x[i]=(b[i]-np.dot(a[i,i+1:],x[i+1:]))/a[i,i]
    return x
