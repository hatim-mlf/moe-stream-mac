import pytest
import numpy as np
from candidate import signed_polygon_area

def test_orientation_and_area():
    ccw=np.array([[0.,0.],[2,0],[0,2]])
    assert signed_polygon_area(ccw)==2.0
    assert signed_polygon_area(ccw[::-1])==-2.0
    assert ccw[1,0]==2
def test_short_and_invalid():
    assert signed_polygon_area(np.empty((0,2)))==0.0
    assert signed_polygon_area([[0,0],[1,0]])==0.0
    with pytest.raises(ValueError): signed_polygon_area([[0,0,1]])
    with pytest.raises(ValueError): signed_polygon_area([[np.inf,0],[0,1],[1,0]])
