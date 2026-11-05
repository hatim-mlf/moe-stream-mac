def merge_adjacent_ranges(ranges):
    items=sorted(ranges); out=[]
    for start,end in items:
        if out and start<=out[-1][1]: out[-1]=(out[-1][0],max(out[-1][1],end))
        else: out.append((start,end))
    return out
