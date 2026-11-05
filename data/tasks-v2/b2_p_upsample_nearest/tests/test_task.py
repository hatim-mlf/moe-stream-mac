import numpy as np
from candidate import upsample_nearest_2x_nhwc
def test_spatial_repeat_dtype_and_copy():
    a=np.array([[[[1],[2]],[[3],[4]]]],dtype=np.int16)
    got=upsample_nearest_2x_nhwc(a)
    assert got.dtype==np.int16 and got.shape==(1,4,4,1)
    assert np.array_equal(got[0,:,:,0],[[1,1,2,2],[1,1,2,2],[3,3,4,4],[3,3,4,4]])
    got[0,0,0,0]=-1
    assert a[0,0,0,0]==1
    assert upsample_nearest_2x_nhwc(np.empty((0,2,3,1))).shape==(0,4,6,1)
def test_invalid_rank():
    import pytest
    with pytest.raises(ValueError): upsample_nearest_2x_nhwc([1,2])
