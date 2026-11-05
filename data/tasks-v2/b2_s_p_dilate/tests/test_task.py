import numpy as np
from candidate import dilate_mask_1d
def test_dilation_edges_and_empty():
    assert np.array_equal(dilate_mask_1d([0,1,0,0,1],1),[True,True,True,True,True])
    assert np.array_equal(dilate_mask_1d([0,0,1,0],0),[False,False,True,False])
    assert dilate_mask_1d([],4).dtype==bool
def test_invalid():
    import pytest
    with pytest.raises(ValueError): dilate_mask_1d([[1]],1)
    with pytest.raises(ValueError): dilate_mask_1d([1],True)
    with pytest.raises(ValueError): dilate_mask_1d([1],-1)
