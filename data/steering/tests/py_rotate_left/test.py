from candidate import rotate_left
source = [1, 2, 3, 4, 5]
assert rotate_left(source, 2) == [3, 4, 5, 1, 2]
assert rotate_left(source, -1) == [5, 1, 2, 3, 4]
assert rotate_left([], 9) == []
assert source == [1, 2, 3, 4, 5]
