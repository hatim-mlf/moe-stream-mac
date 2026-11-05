import pytest
import numpy as np
from candidate import weighted_rms
def test_weighted_value():
    assert np.isclose(weighted_rms([3,4],[3,1]),np.sqrt(43/4))
    with pytest.raises(ValueError): weighted_rms([1],[0])
    assert np.isclose(weighted_rms([1e300,1e300],[1e308,1e308]),1e300)
def test_validation():
    for x,w in [([1],[1,2]),([np.inf],[1]),([1],[-1]),([],[])]:
        with pytest.raises(ValueError): weighted_rms(x,w)
