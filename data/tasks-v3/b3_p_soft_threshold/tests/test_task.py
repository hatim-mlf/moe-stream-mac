import pytest
import numpy as np
from candidate import soft_threshold_array
def test_values_and_copy():
    x=np.array([-3.,-1.,0.,2.])
    got=soft_threshold_array(x,1.)
    assert np.array_equal(got,[-2,0,0,1]) and got.dtype==np.float64
    assert np.array_equal(x,[-3,-1,0,2])
def test_invalid():
    with pytest.raises(ValueError): soft_threshold_array([1],-1)
    with pytest.raises(ValueError): soft_threshold_array([np.inf],1)
