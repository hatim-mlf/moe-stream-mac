import numpy as np
def block_mean_2d(x,height,width):
 a=np.asarray(x,dtype=float)
 if a.ndim!=2 or height<1 or width<1: raise ValueError()
 return a[:a.shape[0]//height*height,:a.shape[1]//width*width].reshape(-1,height,a.shape[1]//width,width).mean((1,3))
