import pytest
import numpy as np
from candidate import linear_resample_1d
def test_linear_and_clamped():
    q=np.array([[-1.,0.5,3.]])
    got=linear_resample_1d(q,[0,2],[1,5])
    assert got.shape==(1,3) and np.allclose(got,[[1,2,5]])
    assert linear_resample_1d([], [0,1], [2,4]).shape==(0,)

def test_invalid():
    with pytest.raises(ValueError): linear_resample_1d([0],[0,0],[1,2])
    with pytest.raises(ValueError): linear_resample_1d([np.nan],[0,1],[1,2])
