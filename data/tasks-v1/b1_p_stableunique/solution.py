def stable_unique_values(values):
    seen=set()
    result=[]
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
