import numpy as np
def diagonal_band_mask(rows,cols,offset,radius):
 if any(not isinstance(v,(int,np.integer)) or isinstance(v,(bool,np.bool_)) for v in (rows,cols,offset,radius)) or rows<0 or cols<0 or radius<0: raise ValueError()
 r=np.arange(rows)[:,None]; c=np.arange(cols)[None,:]; return np.abs((r-c)-offset)<=radius
