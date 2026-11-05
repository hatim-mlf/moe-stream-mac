import pytest
from candidate import format_percent_1dp

def test_format_percent():
    assert format_percent_1dp(0.125)=='12.5%'
    assert format_percent_1dp(-0.034)=='-3.4%'
    with pytest.raises(ValueError): format_percent_1dp(True)
    with pytest.raises(ValueError): format_percent_1dp(float('inf'))
