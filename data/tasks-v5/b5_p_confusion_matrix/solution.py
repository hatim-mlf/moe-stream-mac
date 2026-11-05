import numpy as np

def confusion_matrix_counts(truth,predicted,classes):
    y=np.asarray(truth); p=np.asarray(predicted)
    valid=isinstance(classes,(int,np.integer)) and not isinstance(classes,(bool,np.bool_))
    if y.ndim==1 and p.ndim==1 and y.size==0 and p.size==0:
        y=np.empty(0,dtype=np.int64); p=np.empty(0,dtype=np.int64)
    if y.ndim!=1 or p.ndim!=1 or not np.issubdtype(y.dtype,np.integer) or not np.issubdtype(p.dtype,np.integer) or not valid or classes<=0 or len(y)!=len(p): raise ValueError
    if np.any(y<0) or np.any(y>=classes) or np.any(p<0) or np.any(p>=classes): raise ValueError
    out=np.zeros((int(classes),int(classes)),dtype=np.int64)
    np.add.at(out,(y.astype(np.int64),p.astype(np.int64)),1)
    return out
