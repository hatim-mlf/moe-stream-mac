import numpy as np

def hann_window(length,periodic=False):
    n=int(length)
    if n==0:return np.empty(0)
    if n==1:return np.ones(1)
    return .5-.5*np.cos(2*np.pi*np.arange(n)/(n-1))
