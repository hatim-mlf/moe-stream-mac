from candidate import extend_to
source = [1, 2, 3]
assert extend_to(source, 5, 0) == [1, 2, 3, 0, 0]
assert extend_to(source, 2, 0) == [1, 2, 3]
assert source == [1, 2, 3]
try:
    extend_to([], -1, None)
except ValueError:
    pass
else:
    raise AssertionError('negative length must fail')
