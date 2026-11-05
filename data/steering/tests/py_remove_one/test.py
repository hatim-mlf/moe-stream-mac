from candidate import remove_one
source = [4, 2, 4, 4, 7]
assert remove_one(source, 4) == [2, 4, 4, 7]
assert remove_one(source, 9) == source
assert source == [4, 2, 4, 4, 7]
