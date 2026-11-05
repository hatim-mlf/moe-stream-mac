from candidate import duration_hms
assert duration_hms('09:00:00', '09:01:05') == 65
assert duration_hms('10:00:00', '09:59:59') == -1
for bad in ('9:00:00', '24:00:00', '12:60:00', '１２:00:00'):
    try:
        duration_hms(bad, '00:00:00')
    except ValueError:
        pass
    else:
        raise AssertionError(bad)
