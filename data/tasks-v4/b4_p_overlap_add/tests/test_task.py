import numpy as np
import pytest
from candidate import overlap_add_frames

def test_overlap_add():
    frames=np.array([[1,2,3],[4,5,6],[7,8,9]])
    assert overlap_add_frames(frames,2,6).tolist()==[1,2,7,5,13,8]
    assert overlap_add_frames(frames,2,0).shape==(0,)
    assert overlap_add_frames(np.empty((0,4)),1,3).tolist()==[0,0,0]
    with pytest.raises(ValueError): overlap_add_frames(frames,True,4)
