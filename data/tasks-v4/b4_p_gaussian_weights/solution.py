import numpy as np

def gaussian_weights(radius, sigma):
    if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError('invalid radius')
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError('invalid sigma')
    offsets = np.arange(-int(radius), int(radius)+1, dtype=np.float64)
    weights = np.exp(-0.5 * (offsets / float(sigma)) ** 2)
    return weights / weights.sum()
