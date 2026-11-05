import numpy as np

def hann_window(length,periodic=False):
    valid=isinstance(length,(int,np.integer)) and not isinstance(length,(bool,np.bool_))
    if not valid or length<0 or not isinstance(periodic,(bool,np.bool_)): raise ValueError
    n=int(length)
    if n==0:return np.empty(0,dtype=np.float64)
    if not periodic and n==1:return np.ones(1,dtype=np.float64)
    den=n if periodic else n-1
    i=np.arange(n,dtype=np.float64)
    return 0.5-0.5*np.cos(2*np.pi*i/den)
