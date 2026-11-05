import numpy as np
def points_in_polygon(points,polygon):
    p=np.asarray(points,dtype=float); v=np.asarray(polygon,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or v.ndim!=2 or v.shape[1]!=2 or len(v)<3: raise ValueError()
    out=np.zeros(len(p),bool)
    for q,(x,y) in enumerate(p):
        inside=False
        for i in range(len(v)):
            x1,y1=v[i]; x2,y2=v[(i+1)%len(v)]
            if (y1>y)!=(y2>y) and x<x1+(y-y1)*(x2-x1)/(y2-y1): inside=not inside
        out[q]=inside
    return out
