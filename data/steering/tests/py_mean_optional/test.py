from candidate import mean_optional
assert mean_optional([]) is None
assert mean_optional([1.0, 2.0, 6.0]) == 3.0
assert abs(mean_optional([1e16, 1.0, -1e16]) - (1.0 / 3.0)) < 1e-15
