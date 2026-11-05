import numpy as np
import pytest
from candidate import reflect_pad_1d

def test_pad():
    x=np.array([1,2,3]); y=reflect_pad_1d(x,2,1)
    np.testing.assert_array_equal(y,[2,1,1,2,3,3])
    assert y.dtype==x.dtype and not np.shares_memory(x,y)
    assert reflect_pad_1d([],0,0).shape==(0,)
    with pytest.raises(ValueError): reflect_pad_1d([],1,0)
