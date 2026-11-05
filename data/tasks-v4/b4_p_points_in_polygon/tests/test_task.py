import numpy as np
import pytest
from candidate import points_in_polygon

def test_polygon():
    poly=[[0,0],[4,0],[4,3],[0,3]]
    pts=[[2,2],[5,2],[0,1],[4,3],[2,0]]
    assert points_in_polygon(pts,poly).tolist()==[True,False,True,True,True]
    assert points_in_polygon(np.empty((0,2)),poly).shape==(0,)
    with pytest.raises(ValueError): points_in_polygon([[0,0]],[[0,0],[1,1]])
