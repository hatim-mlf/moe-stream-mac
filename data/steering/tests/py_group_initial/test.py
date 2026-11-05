from candidate import group_initial
items = ['Bee', '', 'apple', 'boat', 'Apricot']
result = group_initial(items)
assert list(result) == ['b', 'a']
assert result == {'b': ['Bee', 'boat'], 'a': ['apple', 'Apricot']}
assert items == ['Bee', '', 'apple', 'boat', 'Apricot']
