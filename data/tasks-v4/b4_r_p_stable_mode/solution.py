def stable_mode(values):
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        raise ValueError('values must not be empty')
    return max(counts, key=counts.get)
