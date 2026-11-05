import numpy as np

def minmax_scale_columns(x):
    a=np.asarray(x,dtype=float); lo=a.min(axis=1,keepdims=True); span=a.max(axis=1,keepdims=True)-lo
    return np.divide(a-lo,span,out=np.zeros_like(a),where=span!=0)
