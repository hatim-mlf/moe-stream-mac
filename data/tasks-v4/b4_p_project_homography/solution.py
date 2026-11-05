import numpy as np

def project_homography2d(points, matrix):
    p = np.asarray(points, dtype=np.float64)
    m = np.asarray(matrix, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or m.shape != (3, 3):
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(m)):
        raise ValueError('values must be finite')
    homogeneous = np.column_stack((p, np.ones(p.shape[0]))) @ m.T
    denominator = homogeneous[:, 2]
    if np.any(denominator == 0):
        raise ValueError('zero homogeneous denominator')
    return homogeneous[:, :2] / denominator[:, None]
