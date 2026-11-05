import numpy as np
def pad_axis_to_block(array,block,axis=-1,value=0):
    a=np.asarray(array)
    extra=(-a.shape[axis])%block
    widths=[(0,0)]*a.ndim; widths[axis]=(0,extra)
    return np.pad(a,widths,constant_values=value)
