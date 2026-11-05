import numpy as np
def true_spans(mask):
    a=np.asarray(mask,dtype=bool)
    return [(i,i+1) for i,x in enumerate(a) if x]
