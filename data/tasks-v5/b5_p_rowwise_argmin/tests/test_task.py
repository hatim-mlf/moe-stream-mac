import numpy as np
import pytest
from candidate import rowwise_argmin

def test_argmin():
    np.testing.assert_array_equal(rowwise_argmin([[2,1,1],[-3,0,4]]),[1,0])
    assert rowwise_argmin(np.empty((0,3))).shape==(0,)
    with pytest.raises(ValueError): rowwise_argmin(np.empty((2,0)))
