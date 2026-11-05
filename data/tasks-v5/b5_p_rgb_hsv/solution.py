import numpy as np

def rgb_to_hsv_unit(rgb):
    a=np.asarray(rgb,dtype=np.float64)
    if a.ndim<1 or a.shape[-1]!=3 or not np.isfinite(a).all() or np.any((a<0)|(a>1)): raise ValueError
    flat=a.reshape(-1,3); out=np.zeros_like(flat)
    for i,(r,g,b) in enumerate(flat):
        mx=max(r,g,b); mn=min(r,g,b); d=mx-mn; h=0.0
        if d:
            if mx==r: h=((g-b)/d)%6.0
            elif mx==g: h=(b-r)/d+2.0
            else: h=(r-g)/d+4.0
            h/=6.0
        out[i]=(h,d/mx if mx else 0.0,mx)
    return out.reshape(a.shape)
