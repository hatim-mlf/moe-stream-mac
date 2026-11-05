import numpy as np
def decode_morton2d(codes, bits=16):
    a = np.asarray(codes)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError("codes must be an integer vector")
    if not isinstance(bits, (int, np.integer)) or isinstance(bits, (bool, np.bool_)) or not 1 <= bits <= 16:
        raise ValueError("bits must be from 1 through 16")
    if a.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if np.any(a < 0) or np.any(a >= (1 << (2 * int(bits)))):
        raise ValueError("code out of range")
    out = np.zeros((a.size, 2), dtype=np.int64)
    for bit in range(int(bits)):
        out[:, 0] |= ((a >> (2 * bit)) & 1).astype(np.int64) << bit
        out[:, 1] |= ((a >> (2 * bit + 1)) & 1).astype(np.int64) << bit
    return out
