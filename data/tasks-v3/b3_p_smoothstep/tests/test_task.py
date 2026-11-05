import pytest
import numpy as np
from candidate import smoothstep01
def test_shape_clip_and_copy():
    x=np.array([[-1.,0.,.5,1.,2.]])
    got=smoothstep01(x)
    assert got.shape==x.shape and np.allclose(got,[[0,0,.5,1,1]])
    assert x[0,0]==-1
def test_nonfinite():
    with pytest.raises(ValueError): smoothstep01([np.inf])
