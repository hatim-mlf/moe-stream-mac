from candidate import split_once
assert split_once('one::two::three', '::') == ('one', 'two::three', True)
assert split_once('plain', ':') == ('plain', '', False)
try:
    split_once('x', '')
except ValueError:
    pass
else:
    raise AssertionError('empty separator must fail')
