def longest_equal_run(values):
    items=list(values)
    return max((items.count(x) for x in items),default=0)
