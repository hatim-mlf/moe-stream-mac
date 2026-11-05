import numpy as np
import pytest
from candidate import finite_counts_2d
def test_counts_and_empty():
    np.testing.assert_array_equal(finite_counts_2d([[1,np.nan],[np.inf,3]]),[1,1])
    got=finite_counts_2d(np.empty((0,3)))
    assert got.shape==(0,) and got.dtype==np.int64
def test_rank():
    with pytest.raises(ValueError): finite_counts_2d([1,2])
