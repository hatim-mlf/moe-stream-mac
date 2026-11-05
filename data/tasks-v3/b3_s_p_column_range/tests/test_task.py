import pytest
import numpy as np
from candidate import column_range
def test_ranges():
    assert np.array_equal(column_range([[1,9],[4,2],[3,5]]),[3,7])
    assert column_range(np.empty((2,0))).shape==(0,)
def test_invalid():
    with pytest.raises(ValueError): column_range(np.empty((0,2)))
    with pytest.raises(ValueError): column_range([[np.nan]])
