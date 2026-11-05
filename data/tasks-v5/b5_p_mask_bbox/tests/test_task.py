import numpy as np
import pytest
from candidate import bbox_from_mask

def test_bbox():
    m=np.zeros((4,5),bool);m[1,2]=1;m[3,4]=1
    assert bbox_from_mask(m)==(1,2,4,5)
    assert bbox_from_mask(np.zeros((2,3),bool)) is None
    with pytest.raises(ValueError): bbox_from_mask([[0,1]])
