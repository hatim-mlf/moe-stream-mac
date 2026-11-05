import numpy as np
def copy_prefix_rows(array,count):
 a=np.asarray(array)
 if a.ndim!=2 or not isinstance(count,(int,np.integer)) or isinstance(count,(bool,np.bool_)): raise ValueError()
 return a[:int(count),:]
