import numpy as np

def nms_xyxy(boxes, scores, threshold):
    b = np.asarray(boxes, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if b.ndim != 2 or b.shape[1] != 4 or s.shape != (b.shape[0],):
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(s)) or not np.isfinite(threshold):
        raise ValueError('values must be finite')
    if np.any(b[:, 2] < b[:, 0]) or np.any(b[:, 3] < b[:, 1]) or not 0 <= threshold <= 1:
        raise ValueError('invalid box or threshold')
    order = np.lexsort((np.arange(len(s)), -s))
    kept = []
    while len(order):
        i = int(order[0]); kept.append(i)
        rest = []
        for j in order[1:]:
            x1, y1 = max(b[i,0],b[j,0]), max(b[i,1],b[j,1])
            x2, y2 = min(b[i,2],b[j,2]), min(b[i,3],b[j,3])
            inter = max(0.0,x2-x1) * max(0.0,y2-y1)
            area_i = (b[i,2]-b[i,0]) * (b[i,3]-b[i,1])
            area_j = (b[j,2]-b[j,0]) * (b[j,3]-b[j,1])
            union = area_i + area_j - inter
            iou = inter / union if union > 0 else 0.0
            if iou <= threshold:
                rest.append(int(j))
        order = np.asarray(rest, dtype=np.int64)
    return np.asarray(kept, dtype=np.int64)
