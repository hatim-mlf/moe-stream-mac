def merge_adjacent_ranges(ranges):
    items=[]
    for pair in ranges:
        if len(pair)!=2:
            raise ValueError("range must have two endpoints")
        start,end=pair
        if not isinstance(start,int) or not isinstance(end,int) or start>end:
            raise ValueError("invalid range")
        items.append((start,end))
    items.sort()
    result=[]
    for start,end in items:
        if result and start<=result[-1][1]+1:
            result[-1]=(result[-1][0],max(result[-1][1],end))
        else:
            result.append((start,end))
    return result
