def edit_distance(source, target):
    previous = list(range(len(target) + 1))
    for i, left in enumerate(source, 1):
        current = [i]
        for j, right in enumerate(target, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(left == right)))
        previous = current
    return previous[-1]
