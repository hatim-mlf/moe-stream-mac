import numpy as np
import pytest
from candidate import binary_roc_auc

def test_auc():
    assert binary_roc_auc([0.1,0.4,0.35,0.8],[0,0,1,1]) == 0.75
    assert binary_roc_auc([1,1],[0,1]) == 0.5
    assert binary_roc_auc([3,2,1],[1,0,0]) == 1.0
    with pytest.raises(ValueError): binary_roc_auc([1,2],[1,1])
