def balanced_slices(length,parts):
    if not isinstance(length,int) or not isinstance(parts,int) or length<0 or parts<=0:
        raise ValueError("invalid length or part count")
    base,extra=divmod(length,parts)
    result=[]
    start=0
    for i in range(parts):
        stop=start+base+(1 if i<extra else 0)
        result.append((start,stop))
        start=stop
    return result
