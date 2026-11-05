import pytest
import numpy as np
from candidate import polyline_length2d
def test_lengths():
    assert polyline_length2d([[0,0],[3,4],[3,0]])==9.0
    assert polyline_length2d(np.empty((0,2)))==0.0
    assert isinstance(polyline_length2d([[1,2]]),float)
def test_invalid():
    with pytest.raises(ValueError): polyline_length2d([[0,0,0]])
    with pytest.raises(ValueError): polyline_length2d([[np.inf,0]])
