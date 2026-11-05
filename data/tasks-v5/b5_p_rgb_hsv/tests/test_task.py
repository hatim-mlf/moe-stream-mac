import numpy as np
import pytest
from candidate import rgb_to_hsv_unit

def test_colors():
    x=np.array([[1,0,0],[0,1,0],[0,0,1],[.5,.5,.5]])
    y=rgb_to_hsv_unit(x)
    np.testing.assert_allclose(y,[[0,1,1],[1/3,1,1],[2/3,1,1],[0,0,.5]],atol=1e-14)
    assert y.shape==x.shape and y.dtype==np.float64 and not np.shares_memory(x,y)
    with pytest.raises(ValueError): rgb_to_hsv_unit([[1,0,2]])
