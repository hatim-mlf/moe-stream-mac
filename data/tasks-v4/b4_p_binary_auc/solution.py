import numpy as np

def binary_roc_auc(scores, labels):
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels)
    if s.ndim != 1 or y.ndim != 1 or y.size != s.size or y.dtype.kind not in 'biu':
        raise ValueError('invalid shapes or label dtype')
    if not np.all(np.isfinite(s)) or np.any((y != 0) & (y != 1)):
        raise ValueError('invalid values')
    positives = y.astype(bool)
    p = int(np.sum(positives)); n = int(y.size - p)
    if p == 0 or n == 0:
        raise ValueError('both classes are required')
    order = np.argsort(s, kind='stable')
    rank_sum = 0.0
    start = 0
    while start < s.size:
        stop = start + 1
        while stop < s.size and s[order[stop]] == s[order[start]]:
            stop += 1
        average_rank = ((start + 1) + stop) / 2.0
        rank_sum += average_rank * int(np.sum(positives[order[start:stop]]))
        start = stop
    return float((rank_sum - p*(p+1)/2.0) / (p*n))
