from candidate import is_even


def test_even_and_odd():
    assert is_even(4) is True
    assert is_even(-3) is False
