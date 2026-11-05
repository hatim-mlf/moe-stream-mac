import numpy as np
import pytest
from candidate import record_high_indices

def test_records():
    assert record_high_indices([3,1,3,4,2,5]).tolist()==[0,3,5]
    assert record_high_indices([]).dtype==np.int64
    with pytest.raises(ValueError): record_high_indices([1,np.inf])
