import numpy as np
def rowwise_inner(left,right):
    a=np.asarray(left); b=np.asarray(right)
    return np.sum(a*b,axis=1)
