import numpy as np
import pytest
from candidate import batched_gather_rows

def test_gather():
    x=np.arange(24,dtype=np.int16).reshape(2,3,4)
    got=batched_gather_rows(x,[[2,0],[1,1]])
    assert got.dtype==np.int16 and got.tolist()==[[x[0,2].tolist(),x[0,0].tolist()],[x[1,1].tolist(),x[1,1].tolist()]]
    got[0,0,0]=-1; assert x[0,2,0] != -1
    assert batched_gather_rows(np.empty((0,3,2)),np.empty((0,0),dtype=int)).shape==(0,0,2)
    with pytest.raises(ValueError): batched_gather_rows(x,[[3],[0]])
