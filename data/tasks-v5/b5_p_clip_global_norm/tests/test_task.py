import numpy as np
import pytest
from candidate import clip_global_norm

def test_global_clip():
    a=np.array([3.,4.]); b=np.array([12.]); out,n=clip_global_norm([a,b],6)
    assert abs(n-13)<1e-12
    np.testing.assert_allclose(out[0],a*6/13); np.testing.assert_allclose(out[1],b*6/13)
    assert not np.shares_memory(out[0],a)
    z,_=clip_global_norm([np.zeros(2)],0); np.testing.assert_array_equal(z[0],[0,0])
    with pytest.raises(ValueError): clip_global_norm([a],-1)
