import pytest
from candidate import weighted_median
def test_weighting_and_boundary():
    assert weighted_median([10,1,4],[1,1,2])==4.0
    assert weighted_median([1,9],[1,1])==1.0
def test_invalid():
    with pytest.raises(ValueError): weighted_median([],[])
    with pytest.raises(ValueError): weighted_median([1,2],[0,0])
    with pytest.raises(ValueError): weighted_median([1],[ -1])
