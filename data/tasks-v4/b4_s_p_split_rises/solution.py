import numpy as np

def split_nondecreasing_runs(values):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'biuf' or not np.all(np.isfinite(a)):
        raise ValueError('expected finite numeric vector')
    if a.size == 0:
        return []
    starts = [0]
    for i in range(1, a.size):
        if a[i] < a[i-1]:
            starts.append(i)
    starts.append(a.size)
    return [a[starts[i]:starts[i+1]].copy() for i in range(len(starts)-1)]
