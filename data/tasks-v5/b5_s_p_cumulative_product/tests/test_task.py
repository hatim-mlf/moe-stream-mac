import numpy as np
import pytest
from candidate import cumulative_product

def test_product():
    np.testing.assert_array_equal(cumulative_product([2,3,0,8]),[2,6,0,0])
    assert cumulative_product([]).dtype==np.float64
    with pytest.raises(ValueError): cumulative_product([1,np.inf])
