import numpy as np
import pytest
from candidate import quantize_symmetric_codes
def test_ties_and_symmetric_limits():
    got=quantize_symmetric_codes([-2.5,-1.5,-0.5,0.5,1.5,2.5,-200],1)
    np.testing.assert_array_equal(got,[-2,-2,0,0,2,2,-127])
    assert got.dtype==np.int8
def test_invalid():
    with pytest.raises(ValueError): quantize_symmetric_codes([np.inf],1)
    with pytest.raises(ValueError): quantize_symmetric_codes([1],0)
