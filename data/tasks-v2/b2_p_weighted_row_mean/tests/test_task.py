import numpy as np
from candidate import weighted_rows_mean
def test_weighting():
    got=weighted_rows_mean([[1,3],[5,7],[9,11]],[1,3,0])
    assert got.dtype==np.float64 and np.allclose(got,[4,6])
    assert weighted_rows_mean(np.empty((2,0)),[1,2]).shape==(0,)
def test_invalid():
    import pytest
    with pytest.raises(ValueError): weighted_rows_mean([[1],[2]],[0,0])
    with pytest.raises(ValueError): weighted_rows_mean([[1],[2]],[1])
    with pytest.raises(ValueError): weighted_rows_mean([[1],[2]],[-1,2])
