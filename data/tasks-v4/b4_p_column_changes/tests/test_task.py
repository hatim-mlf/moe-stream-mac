import numpy as np
import pytest
from candidate import column_change_counts

def test_changes():
    x=np.array([[1,2,0],[1,3,0],[4,3,0],[4,2,1]])
    got=column_change_counts(x)
    assert got.dtype==np.int64 and got.tolist()==[1,2,1]
    assert column_change_counts(np.empty((0,3))).tolist()==[0,0,0]
    assert column_change_counts(np.ones((1,2))).tolist()==[0,0]
    with pytest.raises(ValueError): column_change_counts([1,2])
