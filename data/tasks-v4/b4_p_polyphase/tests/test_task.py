import numpy as np
import pytest
from candidate import polyphase_decompose

def test_polyphase():
    x=np.array([0,1,2,3,4,5],dtype=np.int16)
    parts=polyphase_decompose(x,3)
    assert [p.tolist() for p in parts]==[[0,3],[1,4],[2,5]]
    assert all(p.dtype==np.int16 for p in parts)
    parts[0][0]=99; assert x[0]==0
    assert len(polyphase_decompose([],2))==2
    with pytest.raises(ValueError): polyphase_decompose([1,2,3],2)
