import numpy as np

def points_in_polygon(points, polygon):
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(polygon, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2 or v.ndim != 2 or v.shape[1] != 2 or v.shape[0] < 3:
        raise ValueError('invalid shapes')
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(v)):
        raise ValueError('values must be finite')
    result = np.zeros(p.shape[0], dtype=bool)
    for q, (px, py) in enumerate(p):
        inside = False
        for i in range(len(v)):
            x1, y1 = v[i]; x2, y2 = v[(i+1) % len(v)]
            cross = (px-x1)*(y2-y1) - (py-y1)*(x2-x1)
            if cross == 0 and min(x1,x2) <= px <= max(x1,x2) and min(y1,y2) <= py <= max(y1,y2):
                inside = True
                break
            if (y1 > py) != (y2 > py) and px < x1 + (py-y1)*(x2-x1)/(y2-y1):
                inside = not inside
        result[q] = inside
    return result
