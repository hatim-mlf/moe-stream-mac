import numpy as np

def lower_bound_index(values, target):
    a = np.asarray(values, dtype=np.float64)
    try:
        t = float(target)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('invalid target') from exc
    if a.ndim != 1 or not np.all(np.isfinite(a)) or not np.isfinite(t) or np.any(np.diff(a) < 0):
        raise ValueError('values must be sorted and finite')
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < t:
            lo = mid + 1
        else:
            hi = mid
    return int(lo)
