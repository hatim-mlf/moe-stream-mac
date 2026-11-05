from candidate import even_pairs
result = even_pairs(['a', 'b', 'c', 'd', 'e'])
assert result == [(0, 'a'), (2, 'c'), (4, 'e')]
assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in result)
assert even_pairs([]) == []
