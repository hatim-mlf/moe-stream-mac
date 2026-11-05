import numpy as np
import pytest
from candidate import expand_counts
def test_repeat_and_dtype():
    got=expand_counts(np.array([4,8,9],dtype=np.int16),[2,0,3])
    np.testing.assert_array_equal(got,[4,4,9,9,9])
    assert got.dtype==np.int16
def test_empty_and_invalid():
    assert expand_counts([],[]).size==0
    with pytest.raises(ValueError): expand_counts([1],[1,2])
    with pytest.raises(ValueError): expand_counts([1],[-1])
    with pytest.raises(ValueError): expand_counts([[1],[2]],[1,1])
