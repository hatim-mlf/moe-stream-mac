from candidate import count_prefix
assert count_prefix(['oak', 'oar', 'Oak', 'maple'], 'oa') == 2
assert count_prefix(['x', 'y'], '') == 2
assert count_prefix([], 'a') == 0
