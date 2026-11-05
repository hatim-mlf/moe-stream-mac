import numpy as np

def block_mean_2d(x, height, width):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=2 or not np.isfinite(a).all() or isinstance(height,bool) or isinstance(width,bool) or not isinstance(height,(int,np.integer)) or not isinstance(width,(int,np.integer)) or height<=0 or width<=0:
        raise ValueError("invalid input")
    r=a.shape[0]//height*height; c=a.shape[1]//width*width
    if not r or not c: return np.empty((r//height,c//width),dtype=np.float64)
    return a[:r,:c].reshape(r//height,height,c//width,width).mean(axis=(1,3))
