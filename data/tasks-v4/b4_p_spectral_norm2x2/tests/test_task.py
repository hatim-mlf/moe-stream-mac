import numpy as np
import pytest
from candidate import spectral_norm_2x2

def test_spectral_norm():
    x=np.array([[[3,0],[0,4]],[[0,-1],[1,0]],[[0,0],[0,0]]],float)
    assert np.allclose(spectral_norm_2x2(x),[4,1,0])
    assert spectral_norm_2x2(np.empty((0,2,2))).shape==(0,)
    with pytest.raises(ValueError): spectral_norm_2x2(np.ones((2,3)))
