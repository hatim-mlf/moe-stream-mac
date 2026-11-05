def split_on_sentinel(values, sentinel):
    groups=[]; current=[]
    for value in values:
        if value==sentinel:
            if current: groups.append(current); current=[]
        else: current.append(value)
    if current: groups.append(current)
    return groups
