from candidate import parse_csv_ints
assert parse_csv_ints(' 8, -2,17 ') == [8, -2, 17]
for text in ('', '1,,2', '  ,3'):
    try:
        parse_csv_ints(text)
    except ValueError:
        pass
    else:
        raise AssertionError(repr(text))
