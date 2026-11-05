import numpy as np
from candidate import column_zscore
def test_population_scale_and_constant_column():
    a=np.array([[1.,4.],[3.,4.],[5.,4.]])
    got=column_zscore(a)
    assert got.dtype==np.float64
    assert np.allclose(got[:,0],[-np.sqrt(1.5),0,np.sqrt(1.5)])
    assert np.array_equal(got[:,1],[0,0,0])
    assert np.array_equal(a,[[1,4],[3,4],[5,4]])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): column_zscore([1,2])
    with pytest.raises(ValueError): column_zscore(np.empty((0,2)))
    with pytest.raises(ValueError): column_zscore([[np.inf]])
