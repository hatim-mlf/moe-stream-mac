def longest_strict_rise(values):
    best = current = 0
    previous = None
    have_previous = False
    for value in values:
        if have_previous and value > previous:
            current += 1
        else:
            current = 1
        if current > best:
            best = current
        previous = value
        have_previous = True
    return best
