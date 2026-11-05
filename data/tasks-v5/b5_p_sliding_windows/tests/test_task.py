import numpy as np
import pytest
from candidate import sliding_windows_1d

def test_windows():
    a=np.arange(6,dtype=np.int16); w=sliding_windows_1d(a,3,2)
    np.testing.assert_array_equal(w,[[0,1,2],[2,3,4]])
    assert w.dtype==a.dtype and not np.shares_memory(w,a)
    assert sliding_windows_1d(a,8).shape==(0,8)
    with pytest.raises(ValueError): sliding_windows_1d(a,True)
