import numpy as np
from candidate import argmax_coords
def test_row_major_ties():
    assert argmax_coords([[4,9,1],[9,2,9]])==(0,1)
    assert argmax_coords([[-3]])==(0,0)
def test_invalid():
    import pytest
    with pytest.raises(ValueError): argmax_coords([])
    with pytest.raises(ValueError): argmax_coords([[1,np.nan]])
    with pytest.raises(ValueError): argmax_coords([1,2])
