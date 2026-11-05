import pytest
from candidate import format_duration
def test_format():
    assert format_duration(0)=="0:00:00"
    assert format_duration(3661)=="1:01:01"
    assert format_duration(360000)=="100:00:00"
def test_invalid():
    with pytest.raises(ValueError): format_duration(-1)
    with pytest.raises(ValueError): format_duration(True)
