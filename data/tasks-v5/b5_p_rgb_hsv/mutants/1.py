import numpy as np

def rgb_to_hsv_unit(rgb):
    a=np.asarray(rgb,dtype=float)
    if a.shape[-1]!=3: raise ValueError
    flat=a.reshape(-1,3); out=np.zeros_like(flat)
    for i,(r,g,b) in enumerate(flat):
        mx=max(r,g,b); mn=min(r,g,b); d=mx-mn
        if d:
            h=((g-b)/d)%6 if mx==r else ((b-r)/d+2 if mx==g else (r-g)/d+4)
            out[i]=[h,d/mx,mx]
        else: out[i]=[0,0,mx]
    return out.reshape(a.shape)
