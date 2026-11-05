import numpy as np
import pytest
from candidate import readonly_c_copy

def test_copy():
    x=np.arange(6).reshape(2,3)[:,::2]; y=readonly_c_copy(x)
    assert y.flags.c_contiguous and not y.flags.writeable and not np.shares_memory(x,y)
    assert x.flags.writeable
    with pytest.raises(ValueError): y[0,0]=8
    z=np.arange(4); q=readonly_c_copy(z)
    assert not np.shares_memory(z,q) and z.flags.writeable
