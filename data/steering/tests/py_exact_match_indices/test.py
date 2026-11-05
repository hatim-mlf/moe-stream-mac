from candidate import exact_match_indices
assert exact_match_indices(['red', 'dark red', 'red', 'blue'], 'red') == [0, 2]
assert exact_match_indices(['a', 'b'], 'z') == []
