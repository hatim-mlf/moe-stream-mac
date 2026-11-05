import numpy as np
import pytest
from candidate import gaussian_weights

def test_gaussian():
    w=gaussian_weights(2,1.0)
    assert w.dtype==np.float64 and w.shape==(5,) and np.isclose(w.sum(),1)
    assert np.allclose(w,w[::-1]) and w[2]>w[1]>w[0]
    assert gaussian_weights(0,3).tolist()==[1.0]
    with pytest.raises(ValueError): gaussian_weights(True,1)
