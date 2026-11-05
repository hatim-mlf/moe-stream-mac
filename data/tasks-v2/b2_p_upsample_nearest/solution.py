import numpy as np
def upsample_nearest_2x_nhwc(x):
    a = np.asarray(x)
    if a.ndim != 4:
        raise ValueError("expected NHWC array")
    return np.repeat(np.repeat(a, 2, axis=1), 2, axis=2)
