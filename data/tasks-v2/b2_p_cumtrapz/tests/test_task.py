import numpy as np
from candidate import cumulative_trapezoid
def test_irregular_grid_and_short_inputs():
    got=cumulative_trapezoid([1,3,2],[0,2,5])
    assert np.allclose(got,[0,4,11.5]) and got.dtype==np.float64
    assert np.array_equal(cumulative_trapezoid([],[]),np.array([],dtype=np.float64))
    assert np.array_equal(cumulative_trapezoid([7],[4]),[0])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): cumulative_trapezoid([1,2],[0])
    with pytest.raises(ValueError): cumulative_trapezoid([1,2],[1,1])
    with pytest.raises(ValueError): cumulative_trapezoid([np.nan],[0])
