import numpy as np

def extract_patches_2d(x,patch_h,patch_w,step_y=1,step_x=1):
    a=np.asarray(x,dtype=np.float64)
    valid=lambda z:isinstance(z,(int,np.integer)) and not isinstance(z,(bool,np.bool_))
    dims=(patch_h,patch_w,step_y,step_x)
    if a.ndim!=2 or not np.isfinite(a).all() or any(not valid(v) or v<=0 for v in dims): raise ValueError
    ph,pw,sy,sx=map(int,dims); nr=max(0,1+(a.shape[0]-ph)//sy); nc=max(0,1+(a.shape[1]-pw)//sx)
    out=np.empty((nr*nc,ph,pw),dtype=np.float64); k=0
    for r in range(0,max(a.shape[0]-ph+1,0),sy):
        for c in range(0,max(a.shape[1]-pw+1,0),sx): out[k]=a[r:r+ph,c:c+pw]; k+=1
    return out
