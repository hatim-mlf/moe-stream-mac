import pytest
from candidate import parse_env_lines
def test_comments_duplicates_and_equals_in_value():
    assert parse_env_lines(" #skip\nTOKEN=first\nTOKEN=second\nURL=a=b\n") == {"TOKEN":"first","URL":"a=b"}
def test_invalid():
    with pytest.raises(ValueError): parse_env_lines("nope")
    with pytest.raises(ValueError): parse_env_lines("bad-key=x")
