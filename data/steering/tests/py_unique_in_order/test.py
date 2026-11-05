from candidate import unique_in_order
source = ['pear', 'plum', 'pear', 'lime', 'plum']
result = unique_in_order(source)
assert result == ['pear', 'plum', 'lime']
assert isinstance(result, list)
assert source == ['pear', 'plum', 'pear', 'lime', 'plum']
