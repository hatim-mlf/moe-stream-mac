import numpy as np
import pytest
from candidate import histogram2d_xy

def test_histogram():
    p=np.array([[0,0],[1,1],[2,2],[1,0],[3,1]],float)
    got=histogram2d_xy(p,[0,1,2],[0,1,2])
    assert got.dtype==np.int64 and got.tolist()==[[1,1],[0,2]]
    assert histogram2d_xy(np.empty((0,2)),[0,1],[0,1]).shape==(1,1)
    with pytest.raises(ValueError): histogram2d_xy([[0,0]],[1,0],[0,1])
