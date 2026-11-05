import numpy as np
import pytest
from candidate import row_l1_normalize

def test_rows():
    x=np.array([[1.,-1.],[0.,0.],[-2.,-2.]])
    y=row_l1_normalize(x)
    np.testing.assert_allclose(y,[[.5,-.5],[0,0],[-.5,-.5]])
    assert y.dtype==np.float64 and not np.shares_memory(x,y)
    with pytest.raises(ValueError): row_l1_normalize([[1,np.inf]])
