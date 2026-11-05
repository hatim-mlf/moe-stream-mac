from candidate import add_one


def test_add_one():
    assert add_one(4) == 5
    assert add_one(-2) == -1
