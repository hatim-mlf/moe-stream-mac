import numpy as np
import pytest
from candidate import pad_axis_to_block
def test_padding_and_copy():
    a=np.array([[1,2,3],[4,5,6]],dtype=np.int16)
    b=pad_axis_to_block(a,4,axis=1,value=-1)
    np.testing.assert_array_equal(b,[[1,2,3,-1],[4,5,6,-1]])
    assert b.dtype==a.dtype and not np.shares_memory(a,b)
    np.testing.assert_array_equal(a,[[1,2,3],[4,5,6]])
def test_no_padding_is_copy():
    a=np.arange(4)
    b=pad_axis_to_block(a,2)
    assert not np.shares_memory(a,b)
def test_invalid():
    with pytest.raises(ValueError): pad_axis_to_block([1],0)
    with pytest.raises(ValueError): pad_axis_to_block(np.array(1),2)
