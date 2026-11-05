import numpy as np

def histogram2d_xy(points, x_edges, y_edges):
    p = np.asarray(points, dtype=np.float64)
    xe = np.asarray(x_edges, dtype=np.float64)
    ye = np.asarray(y_edges, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or xe.ndim != 1 or ye.ndim != 1 or xe.size < 2 or ye.size < 2:
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(xe)) or not np.all(np.isfinite(ye)):
        raise ValueError('values must be finite')
    if np.any(np.diff(xe) <= 0) or np.any(np.diff(ye) <= 0):
        raise ValueError('edges must increase')
    return np.histogram2d(p[:,1], p[:,0], bins=(ye, xe))[0].astype(np.int64)
