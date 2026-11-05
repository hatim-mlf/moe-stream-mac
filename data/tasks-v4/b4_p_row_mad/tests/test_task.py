import numpy as np
import pytest
from candidate import row_mad

def test_mad():
    x=np.array([[1,10],[2,14],[100,18]],dtype=np.int16)
    assert np.allclose(row_mad(x),[1,4])
    assert np.allclose(row_mad(x,axis=1),[4.5,6,41])
    with pytest.raises(ValueError): row_mad(np.empty((0,2)))
    with pytest.raises(ValueError): row_mad(x,True)
