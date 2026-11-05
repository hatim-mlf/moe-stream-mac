import numpy as np

def move_feature_axis(values, axis, destination):
    a=np.asarray(values)
    if a.ndim==0: raise ValueError("values must have at least one axis")
    for value in (axis,destination):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)):
            raise TypeError("axes must be integers")
    src=int(axis); dst=int(destination)
    if src<0: src+=a.ndim
    if dst<0: dst+=a.ndim
    if not 0<=src<a.ndim or not 0<=dst<a.ndim:
        raise ValueError("axis out of range")
    return np.moveaxis(a,src,dst)
