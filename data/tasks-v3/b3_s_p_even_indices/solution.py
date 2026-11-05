def evenly_spaced_indices(length, count):
    if any(isinstance(v,bool) or not isinstance(v,int) for v in (length,count)) or length<0 or count<0 or count>length:
        raise ValueError("invalid dimensions")
    if count==0: return []
    return [(i*length)//count for i in range(count)]
