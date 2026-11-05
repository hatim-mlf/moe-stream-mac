import numpy as np
import pytest
from candidate import masked_softmax_rows
def test_masked_and_stable():
    x=np.array([[1000,1001,-5],[1,2,3]],dtype=np.float32)
    m=np.array([[1,1,0],[0,0,0]],dtype=bool)
    before=x.copy(); got=masked_softmax_rows(x,m)
    np.testing.assert_allclose(got[0,:2],[0.268941421,0.731058579],rtol=1e-7)
    np.testing.assert_array_equal(got[0,2:],0)
    np.testing.assert_array_equal(got[1],0)
    np.testing.assert_array_equal(x,before)
    assert got.dtype==np.float64
def test_shape_error():
    with pytest.raises(ValueError): masked_softmax_rows([[1,2]],[[1]])
