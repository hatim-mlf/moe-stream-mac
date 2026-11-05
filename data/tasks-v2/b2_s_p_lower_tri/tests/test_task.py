import numpy as np
from candidate import lower_triangle_values
def test_order_and_copy():
    a=np.array([[1,2,3],[4,5,6],[7,8,9]],dtype=np.int16)
    got=lower_triangle_values(a)
    assert got.dtype==np.float64 and np.array_equal(got,[1,4,5,7,8,9])
    assert np.array_equal(a,[[1,2,3],[4,5,6],[7,8,9]])
    assert lower_triangle_values(np.empty((0,0))).size==0
def test_invalid():
    import pytest
    with pytest.raises(ValueError): lower_triangle_values([[1,2]])
    with pytest.raises(ValueError): lower_triangle_values([[np.inf]])
