import numpy as np
import pytest
from candidate import nms_xyxy

def test_nms():
    boxes=np.array([[0,0,2,2],[0,0,2,2],[1.5,0,2.5,1],[5,5,5,5]],float)
    scores=np.array([0.8,0.9,0.7,0.6])
    assert nms_xyxy(boxes,scores,0.5).tolist()==[1,2,3]
    assert nms_xyxy(boxes,scores,0.0).tolist()==[1,3]
    with pytest.raises(ValueError): nms_xyxy([[1,0,0,1]],[1],0.5)
