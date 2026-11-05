import numpy as np
def one_hot_matrix(indices,classes):
    a=np.asarray(indices,dtype=int)
    out=np.zeros((len(a),classes),dtype=np.uint8)
    out[np.arange(len(a)),a]=1
    return out
