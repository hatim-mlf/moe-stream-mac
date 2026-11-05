import pytest
from candidate import count_nonblank_lines

def test_lines():
    assert count_nonblank_lines('a\n  \nb\t\n')==2
    assert count_nonblank_lines('')==0
    with pytest.raises(ValueError): count_nonblank_lines(None)
