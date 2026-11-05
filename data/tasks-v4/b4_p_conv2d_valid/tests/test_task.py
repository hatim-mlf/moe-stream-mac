import numpy as np
import pytest
from candidate import conv2d_valid

def test_conv():
    x=np.arange(1,10).reshape(3,3)
    assert np.array_equal(conv2d_valid(x,[[1,0],[0,-1]]),[[-4,-4],[-4,-4]])
    assert conv2d_valid(np.ones((1,2)),np.ones((2,1))).shape==(0,2)
    with pytest.raises(ValueError): conv2d_valid([[1]], [[]])
