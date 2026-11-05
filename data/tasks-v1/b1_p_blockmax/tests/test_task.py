import numpy as np
import pytest
from candidate import block_max_1d
def test_partial_block_and_dtype():
    x=np.array([1,8,2,4,3],dtype=np.int16)
    got=block_max_1d(x,2)
    np.testing.assert_array_equal(got,[8,4,3])
    assert got.dtype==np.int16
def test_empty_and_invalid():
    assert block_max_1d(np.array([],dtype=np.float32),3).dtype==np.float32
    with pytest.raises(ValueError): block_max_1d([[1]],2)
    with pytest.raises(ValueError): block_max_1d([1],0)
