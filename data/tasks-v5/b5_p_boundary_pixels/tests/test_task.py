import numpy as np
import pytest
from candidate import boundary_pixels_4

def test_boundary():
    x=np.ones((3,3),dtype=bool); y=boundary_pixels_4(x)
    np.testing.assert_array_equal(y,[[1,1,1],[1,0,1],[1,1,1]])
    assert y.dtype==np.bool_ and not np.shares_memory(x,y)
    assert boundary_pixels_4(np.empty((0,2),dtype=bool)).shape==(0,2)
    with pytest.raises(ValueError): boundary_pixels_4([[1]])
