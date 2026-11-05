import numpy as np
from candidate import histogram_with_edges
def test_edges_and_outside_values():
    x=np.array([-1,0,0.9,1,1.9,2,3])
    assert np.array_equal(histogram_with_edges(x,[0,1,2]),[2,3])
    assert histogram_with_edges([],[-2,0,2]).dtype==np.int64
    assert np.array_equal(histogram_with_edges([0,2],[0,1,2]),[1,1])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): histogram_with_edges([1],[0])
    with pytest.raises(ValueError): histogram_with_edges([1],[0,2,2])
    with pytest.raises(ValueError): histogram_with_edges([np.nan],[0,2])
