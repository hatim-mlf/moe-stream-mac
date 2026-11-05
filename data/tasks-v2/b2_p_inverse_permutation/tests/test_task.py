import numpy as np
from candidate import inverse_permutation
def test_inverse_and_empty():
    p=np.array([2,0,1],dtype=np.int32)
    got=inverse_permutation(p)
    assert got.dtype==np.int64 and np.array_equal(got,[1,2,0])
    assert inverse_permutation([]).shape==(0,)
    assert np.array_equal(p,[2,0,1])
def test_rejects_bad_permutations():
    import pytest
    for p in ([0,0],[0,2],[-1,0],[1.0,0.0],[[0,1]]):
        with pytest.raises(ValueError): inverse_permutation(p)
