import numpy as np

def row_entropy(probabilities):
    p=np.asarray(probabilities,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or np.any(p<0): raise ValueError("invalid probabilities")
    scale=p.max(axis=1,keepdims=True)
    if np.any(scale<=0): raise ValueError("each row needs positive mass")
    p=p/scale
    p=p/p.sum(axis=1,keepdims=True)
    terms=np.zeros_like(p)
    np.log(p,out=terms,where=p>0)
    return -np.sum(np.where(p>0,p*terms,0.0),axis=1)
