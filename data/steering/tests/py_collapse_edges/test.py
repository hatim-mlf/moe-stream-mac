from candidate import collapse_edges
source = [2, 2, 1, 2, 2, 3, 3]
assert collapse_edges(source) == [2, 1, 2, 3]
assert source == [2, 2, 1, 2, 2, 3, 3]
assert collapse_edges([]) == []
