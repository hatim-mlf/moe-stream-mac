import numpy as np
import pytest
from candidate import cumulative_logsumexp

def test_cum_lse():
    np.testing.assert_allclose(cumulative_logsumexp([1000,1000]),[1000,1000+np.log(2)])
    np.testing.assert_allclose(cumulative_logsumexp([0,0,0]),np.log([1,2,3]))
    assert cumulative_logsumexp([]).dtype==np.float64
    with pytest.raises(ValueError): cumulative_logsumexp([np.inf])
