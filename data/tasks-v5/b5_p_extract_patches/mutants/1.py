import numpy as np

def extract_patches_2d(x,patch_h,patch_w,step_y=1,step_x=1):
    a=np.asarray(x,dtype=float); out=[]
    for r in range(0,a.shape[0]-patch_h+1,step_y):
        for c in range(0,a.shape[1]-patch_w+1,step_x): out.append(a[r:r+patch_h,c:c+patch_w])
    return np.asarray(out)
