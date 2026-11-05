import numpy as np
def histogram2d_xy(points,x_edges,y_edges):
    p=np.asarray(points,dtype=float); x=np.asarray(x_edges,dtype=float); y=np.asarray(y_edges,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or x.ndim!=1 or y.ndim!=1: raise ValueError()
    out=np.zeros((len(y)-1,len(x)-1),dtype=np.int64)
    for px,py in p:
        ix=np.searchsorted(x,px,side='right')-1; iy=np.searchsorted(y,py,side='right')-1
        if 0<=ix<len(x)-1 and 0<=iy<len(y)-1: out[iy,ix]+=1
    return out
