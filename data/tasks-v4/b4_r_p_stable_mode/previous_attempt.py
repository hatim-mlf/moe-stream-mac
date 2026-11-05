def stable_mode(values):
    data=list(values)
    if not data: raise ValueError('empty')
    counts={}
    for value in data: counts[value]=counts.get(value,0)+1
    return max(counts,key=lambda value:(counts[value],value))
