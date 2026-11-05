import pytest
import numpy as np
from candidate import apply_integer_lut
def test_lookup_dtype_shape():
    lut=np.array([2,4,8],dtype=np.int16)
    got=apply_integer_lut(np.array([[2,0],[1,1]],dtype=np.uint8),lut)
    assert got.dtype==np.int16 and np.array_equal(got,[[8,2],[4,4]])
    assert apply_integer_lut(np.array([],dtype=np.int64),lut).shape==(0,)
def test_invalid():
    with pytest.raises(ValueError): apply_integer_lut(np.array([3]),[1,2,3])
    with pytest.raises(ValueError): apply_integer_lut(np.array([1.]),[1,2])
    with pytest.raises(ValueError): apply_integer_lut(np.array([0]),[])
