def split_by_budget(costs, limit):
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError("limit must be positive")
    groups = []
    current = []
    total = 0
    for cost in costs:
        if not isinstance(cost, int) or isinstance(cost, bool) or cost < 0 or cost > limit:
            raise ValueError("invalid cost")
        if current and total + cost > limit:
            groups.append(current)
            current = []
            total = 0
        current.append(cost)
        total += cost
    if current:
        groups.append(current)
    return groups
