import numpy as np
def lower_triangle_values(matrix):
    a = np.asarray(matrix, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError("expected finite square matrix")
    return a[np.tril_indices(a.shape[0])].copy()
