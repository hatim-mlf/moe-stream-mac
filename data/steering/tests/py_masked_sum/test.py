from candidate import masked_sum
assert masked_sum([4, 8, 3, 9], [True, False, 1, 0]) == 7
for values, mask in (([1, 2], [True]), ([1], [True, False])):
    try:
        masked_sum(values, mask)
    except ValueError:
        pass
    else:
        raise AssertionError('length mismatch must fail')
