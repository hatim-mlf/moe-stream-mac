import numpy as np
import pytest
from candidate import ema_series
def test_recurrence_and_empty():
    np.testing.assert_allclose(ema_series([2,6,2],0.5),[2,4,3])
    got=ema_series([],0.5)
    assert got.shape==(0,) and got.dtype==np.float64
def test_invalid():
    with pytest.raises(ValueError): ema_series([1,np.inf],0.5)
    with pytest.raises(ValueError): ema_series([1],0)
