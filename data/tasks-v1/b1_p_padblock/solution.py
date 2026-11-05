import numpy as np
def pad_axis_to_block(array, block, axis=-1, value=0):
    a=np.asarray(array)
    if not isinstance(block,(int,np.integer)) or block<=0:
        raise ValueError("block must be positive")
    if a.ndim==0 or axis < -a.ndim or axis >= a.ndim:
        raise ValueError("invalid axis")
    axis %= a.ndim
    extra=(-a.shape[axis])%int(block)
    widths=[(0,0)]*a.ndim
    widths[axis]=(0,extra)
    return np.pad(a,widths,mode="constant",constant_values=value).astype(a.dtype,copy=False)
