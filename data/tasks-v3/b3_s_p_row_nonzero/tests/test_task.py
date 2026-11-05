import pytest
import numpy as np
from candidate import rows_with_nonzero
def test_row_indices():
    assert np.array_equal(rows_with_nonzero([[0,0],[1,0],[0,-2]]),[1,2])
    assert rows_with_nonzero(np.empty((3,0))).shape==(0,)
    assert rows_with_nonzero(np.zeros((0,2))).dtype==np.int64
def test_rank():
    with pytest.raises(ValueError): rows_with_nonzero([1,2])
