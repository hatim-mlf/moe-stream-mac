import pytest
import numpy as np
from candidate import block_mean_2d
def test_blocks_and_dropped_edges():
    x=np.arange(15).reshape(3,5)
    got=block_mean_2d(x,2,2)
    assert got.dtype==np.float64 and np.array_equal(got,[[3,5]])
    assert block_mean_2d([[7]],2,1).shape==(0,1)
    assert x[0,0]==0

def test_validation():
    for args in [([[1]],0,1), ([[1]],1,True), ([[np.inf]],1,1), ([1,2],1,1)]:
        with pytest.raises(ValueError): block_mean_2d(*args)
