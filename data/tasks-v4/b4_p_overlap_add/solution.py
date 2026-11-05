import numpy as np

def overlap_add_frames(frames, hop, length):
    a = np.asarray(frames, dtype=np.float64)
    if a.ndim != 2 or isinstance(hop, (bool, np.bool_)) or not isinstance(hop, (int, np.integer)) or hop <= 0:
        raise ValueError('invalid frames or hop')
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)) or length < 0:
        raise ValueError('invalid length')
    if not np.all(np.isfinite(a)):
        raise ValueError('frames must be finite')
    out = np.zeros(int(length), dtype=np.float64)
    for i, frame in enumerate(a):
        start = i * int(hop)
        if start >= length:
            break
        stop = min(int(length), start + a.shape[1])
        out[start:stop] += frame[:stop-start]
    return out
