import numpy as np

def polyphase_decompose(values, factor):
    a = np.asarray(values)
    if a.ndim != 1 or isinstance(factor, (bool, np.bool_)) or not isinstance(factor, (int, np.integer)) or factor <= 0:
        raise ValueError('invalid vector or factor')
    if a.size % int(factor) != 0:
        raise ValueError('length must be divisible by factor')
    return [a[r::int(factor)].copy() for r in range(int(factor))]
