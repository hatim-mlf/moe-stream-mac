from candidate import insert_after
source = ['x', 'm', 'y', 'm']
assert insert_after(source, 'm', 0) == ['x', 'm', 0, 'y', 'm']
assert insert_after(source, 'z', 0) == ['x', 'm', 'y', 'm', 0]
assert source == ['x', 'm', 'y', 'm']
