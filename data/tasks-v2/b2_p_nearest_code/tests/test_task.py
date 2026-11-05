import numpy as np
from candidate import nearest_codebook
def test_indices_ties_and_empty():
    got=nearest_codebook([0,2,10],[1,3,9])
    assert got.dtype==np.int64 and np.array_equal(got,[0,0,2])
    assert nearest_codebook([], [1]).shape==(0,)
def test_invalid():
    import pytest
    with pytest.raises(ValueError): nearest_codebook([1],[])
    with pytest.raises(ValueError): nearest_codebook([[1]],[1])
    with pytest.raises(ValueError): nearest_codebook([np.inf],[1])
