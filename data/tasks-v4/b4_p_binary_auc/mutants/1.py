import numpy as np
def binary_roc_auc(scores,labels):
    s=np.asarray(scores,dtype=float); y=np.asarray(labels).astype(bool)
    p=s[y]; n=s[~y]
    if not len(p) or not len(n): raise ValueError()
    return float(np.mean(p[:,None] > n[None,:]))
