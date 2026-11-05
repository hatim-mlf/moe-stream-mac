import numpy as np
def argmax_coords(x):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError("expected nonempty finite matrix")
    row, col = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(row), int(col)
