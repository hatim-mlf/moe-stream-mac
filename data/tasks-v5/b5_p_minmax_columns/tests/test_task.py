import numpy as np
import pytest
from candidate import minmax_scale_columns

def test_columns():
    x=np.array([[1,7],[2,7],[4,7]])
    np.testing.assert_allclose(minmax_scale_columns(x),[[0,0],[1/3,0],[1,0]])
    assert minmax_scale_columns(np.empty((0,2))).shape==(0,2)
    with pytest.raises(ValueError): minmax_scale_columns([[np.inf]])
