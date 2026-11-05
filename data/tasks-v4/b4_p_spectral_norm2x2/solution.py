import numpy as np

def spectral_norm_2x2(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim < 2 or a.shape[-2:] != (2, 2) or not np.all(np.isfinite(a)):
        raise ValueError('expected finite 2 by 2 matrices')
    p, q, r, s = a[...,0,0], a[...,0,1], a[...,1,0], a[...,1,1]
    aa = p*p + r*r
    bb = q*q + s*s
    ab = p*q + r*s
    largest = (aa + bb + np.sqrt((aa-bb)**2 + 4*ab*ab)) / 2
    return np.sqrt(np.maximum(largest, 0.0)).astype(np.float64)
