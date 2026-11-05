import numpy as np
import pytest
from candidate import fill_nearest_1d

def test_fill():
    x=np.array([np.nan,2,np.nan,4,np.nan]); y=fill_nearest_1d(x)
    np.testing.assert_array_equal(y,[2,2,2,4,4]); assert np.isnan(x[0])
    np.testing.assert_array_equal(fill_nearest_1d([]),[])
    with pytest.raises(ValueError): fill_nearest_1d([np.nan])
    with pytest.raises(ValueError): fill_nearest_1d([1,np.inf])
