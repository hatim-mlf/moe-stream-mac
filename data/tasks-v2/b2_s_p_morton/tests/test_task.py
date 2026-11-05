import numpy as np
from candidate import decode_morton2d
def test_deinterleave_bits():
    got=decode_morton2d([0,1,2,3,9],2)
    assert got.dtype==np.int64 and np.array_equal(got,[[0,0],[1,0],[0,1],[1,1],[1,2]])
    assert decode_morton2d([],3).shape==(0,2)
def test_invalid():
    import pytest
    with pytest.raises(ValueError): decode_morton2d([-1],2)
    with pytest.raises(ValueError): decode_morton2d([16],2)
    with pytest.raises(ValueError): decode_morton2d([1],True)
    with pytest.raises(ValueError): decode_morton2d([[1]],2)
