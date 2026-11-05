import numpy as np

def solve_tridiagonal(lower, diagonal, upper, rhs):
    lo, d, up, b = (np.asarray(v, dtype=np.float64) for v in (lower, diagonal, upper, rhs))
    n = d.size
    if d.ndim != 1 or n == 0 or b.shape != (n,) or lo.shape != (n-1,) or up.shape != (n-1,):
        raise ValueError('invalid shapes')
    if not all(np.all(np.isfinite(v)) for v in (lo,d,up,b)):
        raise ValueError('values must be finite')
    diag, rhs_work = d.copy(), b.copy()
    for i in range(1, n):
        if diag[i-1] == 0:
            raise ValueError('zero pivot')
        factor = lo[i-1] / diag[i-1]
        diag[i] -= factor * up[i-1]
        rhs_work[i] -= factor * rhs_work[i-1]
    if diag[-1] == 0:
        raise ValueError('zero pivot')
    out = np.empty(n, dtype=np.float64)
    out[-1] = rhs_work[-1] / diag[-1]
    for i in range(n-2, -1, -1):
        if diag[i] == 0:
            raise ValueError('zero pivot')
        out[i] = (rhs_work[i] - up[i] * out[i+1]) / diag[i]
    return out
