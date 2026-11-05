import numpy as np
import pytest
from candidate import rgb_luma
def test_channels_and_leading_shape():
    got=rgb_luma(np.array([[[255,0,0],[0,255,0]]],dtype=np.uint8))
    assert got.shape==(1,2) and got.dtype==np.float64
    np.testing.assert_allclose(got,[[54.213,182.376]])
def test_invalid():
    with pytest.raises(ValueError): rgb_luma([1,2])
    with pytest.raises(ValueError): rgb_luma(1)
