from candidate import partition_size
assert partition_size([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
assert partition_size([], 3) == []
try:
    partition_size([1], 0)
except ValueError:
    pass
else:
    raise AssertionError('size=0 must fail')
