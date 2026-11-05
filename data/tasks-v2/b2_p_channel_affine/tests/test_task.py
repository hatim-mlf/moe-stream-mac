import numpy as np
from candidate import channel_affine
def test_scalar_and_channel_parameters():
    x=np.array([[[1,2],[3,4]]],dtype=np.int16)
    got=channel_affine(x,[2,3],[10,-1])
    assert got.dtype==np.float64 and np.array_equal(got,[[[12,5],[16,11]]])
    assert np.array_equal(channel_affine([[2,3]],2,1),[[5,7]])
    assert np.array_equal(x,[[[1,2],[3,4]]])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): channel_affine(2,[1],[0])
    with pytest.raises(ValueError): channel_affine([[1,2]],[1,2,3],0)
    with pytest.raises(ValueError): channel_affine([[np.inf]],1,0)
