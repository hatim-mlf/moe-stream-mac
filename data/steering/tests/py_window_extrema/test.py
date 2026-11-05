from candidate import window_extrema
result = window_extrema([4, 1, 7, 2], 2)
assert result == ([1, 1, 2], [4, 7, 7])
assert isinstance(result, tuple) and len(result) == 2
assert window_extrema([1], 2) == ([], [])
try:
    window_extrema([], 0)
except ValueError:
    pass
else:
    raise AssertionError('size must be positive')
