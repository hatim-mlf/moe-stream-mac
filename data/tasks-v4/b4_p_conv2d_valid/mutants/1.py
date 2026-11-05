import numpy as np
def conv2d_valid(image,kernel):
    x=np.asarray(image,dtype=float); k=np.asarray(kernel,dtype=float)
    if x.ndim!=2 or k.ndim!=2 or not k.size: raise ValueError()
    h=max(x.shape[0]-k.shape[0]+1,0); w=max(x.shape[1]-k.shape[1]+1,0); out=np.empty((h,w))
    for r in range(h):
        for c in range(w): out[r,c]=np.sum(x[r:r+k.shape[0],c:c+k.shape[1]]*k[::-1,::-1])
    return out
