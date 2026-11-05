import numpy as np
import pytest
from candidate import hann_window

def test_hann():
    np.testing.assert_allclose(hann_window(5),[0,.5,1,.5,0],atol=1e-15)
    np.testing.assert_allclose(hann_window(4,True),[0,.5,1,.5],atol=1e-15)
    np.testing.assert_array_equal(hann_window(1),[1]); assert hann_window(1,True)[0]==0
    with pytest.raises(ValueError): hann_window(True)
