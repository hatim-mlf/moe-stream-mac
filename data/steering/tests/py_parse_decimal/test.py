from candidate import parse_decimal
assert parse_decimal('+17') == 17
assert parse_decimal('-0') == 0
for value in (' 3', '3 ', '1_0', '٣', '+', ''):
    assert parse_decimal(value) is None, repr(value)
assert parse_decimal(None) is None
