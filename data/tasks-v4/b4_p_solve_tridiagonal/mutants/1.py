import numpy as np
def solve_tridiagonal(lower,diagonal,upper,rhs):
    lo,d,up,b=(np.asarray(v,dtype=float) for v in (lower,diagonal,upper,rhs)); n=len(d)
    if lo.shape!=(n-1,) or up.shape!=(n-1,) or b.shape!=(n,): raise ValueError()
    out=np.empty(n); out[0]=b[0]/d[0]
    for i in range(1,n): out[i]=(b[i]-lo[i-1]*out[i-1])/d[i]
    return out
