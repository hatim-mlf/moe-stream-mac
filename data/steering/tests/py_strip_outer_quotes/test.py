from candidate import strip_outer_quotes
assert strip_outer_quotes("'tag'") == 'tag'
assert strip_outer_quotes('"tag"') == 'tag'
assert strip_outer_quotes("''tag''") == "'tag'"
assert strip_outer_quotes("'tag" ) == "'tag"
assert strip_outer_quotes('') == ''
