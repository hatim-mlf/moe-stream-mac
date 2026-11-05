import numpy as np
import pytest
from candidate import one_hot_matrix
def test_rows_and_empty():
    got=one_hot_matrix(np.array([2,0,1]),3)
    np.testing.assert_array_equal(got,[[0,0,1],[1,0,0],[0,1,0]])
    assert got.dtype==np.uint8
    assert one_hot_matrix([],4).shape==(0,4)
def test_invalid():
    with pytest.raises(ValueError): one_hot_matrix([-1],3)
    with pytest.raises(ValueError): one_hot_matrix([1.0],3)
    with pytest.raises(ValueError): one_hot_matrix([0],0)
