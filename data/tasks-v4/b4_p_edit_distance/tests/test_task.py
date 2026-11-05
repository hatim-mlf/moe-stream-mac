from candidate import edit_distance

def test_distance():
    assert edit_distance('', 'abc') == 3
    assert edit_distance('kitten', 'sitting') == 3
    assert edit_distance('é🙂', '🙂') == 1
    assert edit_distance('same', 'same') == 0
