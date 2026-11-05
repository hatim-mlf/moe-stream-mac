from candidate import min_with_index
assert min_with_index([5, 2, 2, 7]) == (1, 2)
assert min_with_index([]) is None
source = [3, 1, 4]
assert min_with_index(source) == (1, 1) and source == [3, 1, 4]
