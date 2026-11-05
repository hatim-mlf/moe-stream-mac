import numpy as np
import pytest
from candidate import last_true_index

def test_last():
    assert last_true_index([False,True,False,True])==3
    assert last_true_index([False,False])==-1
    with pytest.raises(ValueError): last_true_index([0,1])
