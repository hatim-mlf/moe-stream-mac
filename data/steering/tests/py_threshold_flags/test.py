from candidate import threshold_flags
assert threshold_flags([2, 3, 4, 3], 3) == [False, True, True, True]
assert threshold_flags([], 3) == []
