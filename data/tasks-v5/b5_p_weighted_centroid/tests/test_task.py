import numpy as np
import pytest
from candidate import weighted_centroid_2d

def test_centroid():
    assert weighted_centroid_2d([[0,0],[0,4]])==(1.0,1.0)
    assert weighted_centroid_2d([[1,0],[0,1]])==(.5,.5)
    with pytest.raises(ValueError): weighted_centroid_2d([[0,0]])
    with pytest.raises(ValueError): weighted_centroid_2d([[1,-1]])
