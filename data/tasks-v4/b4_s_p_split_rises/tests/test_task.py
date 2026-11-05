import numpy as np
import pytest
from candidate import split_nondecreasing_runs

def test_split():
    parts=split_nondecreasing_runs(np.array([1,2,2,0,3,-1],dtype=np.int16))
    assert [p.tolist() for p in parts]==[[1,2,2],[0,3],[-1]]
    assert all(p.dtype==np.int16 for p in parts)
    assert split_nondecreasing_runs([])==[]
    with pytest.raises(ValueError): split_nondecreasing_runs([[1,2]])
