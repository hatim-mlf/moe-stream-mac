import numpy as np
from candidate import box_sum_valid
def test_valid_sums_and_empty_shape():
    x=np.arange(1,13,dtype=np.int16).reshape(3,4)
    got=box_sum_valid(x,2,2)
    assert got.dtype==np.float64 and np.array_equal(got,[[14,18,22],[30,34,38]])
    assert box_sum_valid(x,4,2).shape==(0,3)
    assert np.array_equal(x,np.arange(1,13).reshape(3,4))
def test_invalid():
    import pytest
    with pytest.raises(ValueError): box_sum_valid([1,2],1,1)
    with pytest.raises(ValueError): box_sum_valid([[np.nan]],1,1)
    with pytest.raises(ValueError): box_sum_valid([[1]],True,1)
    with pytest.raises(ValueError): box_sum_valid([[1]],0,1)
