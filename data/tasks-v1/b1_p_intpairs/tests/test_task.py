import pytest
from candidate import parse_int_pairs
def test_records():
    assert parse_int_pairs(" 2 : -4\n\n+7:0\n")==[(2,-4),(7,0)]
def test_malformed():
    for text in ("1:2:3","a:2","1_0:2","1:"):
        with pytest.raises(ValueError): parse_int_pairs(text)
