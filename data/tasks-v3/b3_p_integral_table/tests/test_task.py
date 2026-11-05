import pytest
import numpy as np
from candidate import integral_table_2d
def test_prefix_rectangle_values():
    x=np.array([[1,2,3],[4,5,6]])
    got=integral_table_2d(x)
    assert got.shape==(3,4) and np.array_equal(got,[[0,0,0,0],[0,1,3,6],[0,5,12,21]])
    assert integral_table_2d(np.empty((0,2))).shape==(1,3)
    assert x[1,2]==6

def test_invalid():
    with pytest.raises(ValueError): integral_table_2d([1,2])
    with pytest.raises(ValueError): integral_table_2d([[np.nan]])
