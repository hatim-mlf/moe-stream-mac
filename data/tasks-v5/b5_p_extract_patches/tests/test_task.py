import numpy as np
import pytest
from candidate import extract_patches_2d

def test_patches():
    x=np.arange(12).reshape(3,4); p=extract_patches_2d(x,2,2,1,2)
    np.testing.assert_array_equal(p,[[[0,1],[4,5]],[[2,3],[6,7]],[[4,5],[8,9]],[[6,7],[10,11]]])
    assert p.dtype==np.float64 and not np.shares_memory(p,x)
    assert extract_patches_2d(x,5,2).shape==(0,5,2)
    with pytest.raises(ValueError): extract_patches_2d(x,True,2)
