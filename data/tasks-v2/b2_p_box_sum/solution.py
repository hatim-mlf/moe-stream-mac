import numpy as np
def box_sum_valid(x, height, width):
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("expected finite matrix")
    for size in (height, width):
        if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)) or size <= 0:
            raise ValueError("window dimensions must be positive integers")
    out_shape = (max(0, a.shape[0] - int(height) + 1), max(0, a.shape[1] - int(width) + 1))
    if height > a.shape[0] or width > a.shape[1]:
        return np.empty(out_shape, dtype=np.float64)
    prefix = np.pad(a, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
    h, w = int(height), int(width)
    return prefix[h:, w:] - prefix[:-h, w:] - prefix[h:, :-w] + prefix[:-h, :-w]
