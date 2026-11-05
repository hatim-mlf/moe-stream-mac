from candidate import clamp_values
source = [-3, 2, 12]
result = clamp_values(source, 0, 8)
assert result == [0, 2, 8]
assert source == [-3, 2, 12]
try:
    clamp_values([1], 5, 2)
except ValueError:
    pass
else:
    raise AssertionError('invalid bounds must fail')
