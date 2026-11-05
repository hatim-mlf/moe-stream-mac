from candidate import format_key
assert format_key('  Two  Words!  ') == 'two--words!'
assert format_key('A_B') == 'a_b'
assert format_key('   ') == ''
