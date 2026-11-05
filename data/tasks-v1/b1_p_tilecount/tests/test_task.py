import numpy as np
import pytest
from candidate import count_true_tiles
def test_partial_tiles():
    a=np.array([[1,0,1],[1,1,0],[0,1,1]],dtype=bool)
    np.testing.assert_array_equal(count_true_tiles(a,2,2),[[3,1],[1,1]])
def test_empty_and_bad():
    assert count_true_tiles(np.zeros((0,4),dtype=bool),2,3).shape==(0,2)
    with pytest.raises(ValueError): count_true_tiles([[1]],1,1)
    with pytest.raises(ValueError): count_true_tiles(np.zeros((2,2),bool),0,1)
