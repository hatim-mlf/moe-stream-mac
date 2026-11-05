import numpy as np
import pytest
from candidate import quantize_unit_u8
def test_rounding_clipping_and_shape():
    got=quantize_unit_u8([-1,0,0.5,1,2],0,1)
    np.testing.assert_array_equal(got,[0,0,128,255,255])
    assert got.dtype==np.uint8 and got.shape==(5,)
def test_reject_bad_values():
    with pytest.raises(ValueError): quantize_unit_u8([1],2,2)
    with pytest.raises(ValueError): quantize_unit_u8([np.inf],0,1)
