def first_duplicate_item(values):
    seen=set(); duplicate=None
    for value in values:
        if value in seen: duplicate=value
        seen.add(value)
    return duplicate
