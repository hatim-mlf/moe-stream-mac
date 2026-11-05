from candidate import map_with_count
source = [1, 2, 3, 4]
result = map_with_count(source, lambda value: value if value % 2 == 0 else value * 10)
assert result == ([10, 2, 30, 4], 2)
assert source == [1, 2, 3, 4]
assert map_with_count([], str) == ([], 0)
