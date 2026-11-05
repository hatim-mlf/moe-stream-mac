import pytest
import numpy as np
from candidate import median_filter_valid
def test_valid_medians():
    assert np.array_equal(median_filter_valid([1,9,3,7],2),[5,6,5])
    assert np.array_equal(median_filter_valid([1,5,2],3),[2])
    assert median_filter_valid([1],2).dtype==np.float64 and median_filter_valid([1],2).size==0
def test_invalid():
    with pytest.raises(ValueError): median_filter_valid([[1,2]],1)
    with pytest.raises(ValueError): median_filter_valid([1],True)
    with pytest.raises(ValueError): median_filter_valid([np.nan],1)
