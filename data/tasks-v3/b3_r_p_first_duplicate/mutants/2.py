def first_duplicate_item(values):
    seen=set()
    for value in values:
        if value not in seen: return value
        seen.add(value)
    return None
