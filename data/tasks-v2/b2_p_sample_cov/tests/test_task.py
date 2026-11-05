import numpy as np
from candidate import sample_covariance
def test_sample_covariance():
    x=np.array([[1,2],[2,4],[3,8]],dtype=np.int16)
    got=sample_covariance(x)
    assert got.dtype==np.float64 and np.allclose(got,[[1,3],[3,9.333333333333]])
    assert sample_covariance(np.ones((2,0))).shape==(0,0)
    assert np.array_equal(x,[[1,2],[2,4],[3,8]])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): sample_covariance([[1,2]])
    with pytest.raises(ValueError): sample_covariance([[1],[np.nan]])
    with pytest.raises(ValueError): sample_covariance([1,2])
