import numpy as np
def smoothstep01(x):
 a=np.asarray(x,dtype=float); return a*a*(3-2*a)
