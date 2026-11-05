def balanced_slices(length,parts):
    if length<0 or parts<=0: raise ValueError
    base=length//parts
    return [(i*base,(i+1)*base if i<parts-1 else length) for i in range(parts)]
