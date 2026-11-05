from candidate import lookup_first
assert lookup_first({'b': 4, 'a': 9}, ['a', 'b']) == (True, 9)
assert lookup_first({'a': None}, ['a']) == (True, None)
assert lookup_first({}, ['x']) == (False, None)
