import numpy as np
from candidate import sliding_dot_valid
def test_valid_correlation():
    assert np.allclose(sliding_dot_valid([1,2,3,4],[1,0,-1]),[-2,-2])
    assert np.allclose(sliding_dot_valid([1,2],[2]),[2,4])
    assert sliding_dot_valid([1],[1,2]).dtype==np.float64 and sliding_dot_valid([1],[1,2]).size==0
def test_invalid():
    import pytest
    with pytest.raises(ValueError): sliding_dot_valid([[1]], [1])
    with pytest.raises(ValueError): sliding_dot_valid([1], [])
    with pytest.raises(ValueError): sliding_dot_valid([np.nan], [1])
