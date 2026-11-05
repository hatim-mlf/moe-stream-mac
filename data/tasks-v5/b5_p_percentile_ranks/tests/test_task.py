import numpy as np
import pytest
from candidate import percentile_ranks

def test_ranks():
    np.testing.assert_allclose(percentile_ranks([2,1,2,4]),[.5,.125,.5,.875])
    np.testing.assert_array_equal(percentile_ranks([]),np.empty(0))
    with pytest.raises(ValueError): percentile_ranks([np.nan])
