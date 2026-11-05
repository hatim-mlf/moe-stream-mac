import math
import numpy as np

def clip_global_norm(tensors,max_norm):
    a=[np.asarray(t,dtype=float) for t in tensors]
    norm=math.sqrt(sum(float(np.sum(x*x)) for x in a))
    return [x*(min(1,max_norm/float(np.linalg.norm(x))) if np.linalg.norm(x) else 1) for x in a],norm
