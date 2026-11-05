import numpy as np
def gaussian_weights(radius,sigma):
    if radius<0 or sigma<=0: raise ValueError()
    x=np.arange(-radius,radius+1,dtype=float)
    return np.exp(-(x/sigma)**2)/np.exp(-(x/sigma)**2).sum()
