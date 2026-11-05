import numpy as np
import pytest
from candidate import rowwise_inner
def test_values_and_dtype():
    got=rowwise_inner([[1,2],[3,4]],[[2,5],[-1,2]])
    np.testing.assert_allclose(got,[12,5])
    assert got.shape==(2,) and got.dtype==np.float64
def test_empty_and_shape_error():
    assert rowwise_inner(np.empty((0,3)),np.empty((0,3))).shape==(0,)
    with pytest.raises(ValueError): rowwise_inner([1,2],[1,2])
    with pytest.raises(ValueError): rowwise_inner([[1]],[[1,2]])
