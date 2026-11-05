import numpy as np
import pytest
from candidate import indices_above_threshold

def test_indices():
    np.testing.assert_array_equal(indices_above_threshold([1,2,2,3],2),[3])
    assert indices_above_threshold([],0).dtype==np.int64
    with pytest.raises(ValueError): indices_above_threshold([np.nan],0)
