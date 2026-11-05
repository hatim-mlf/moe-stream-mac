import numpy as np
import pytest
from candidate import confusion_matrix_counts

def test_confusion():
    np.testing.assert_array_equal(confusion_matrix_counts([0,0,1,2],[0,1,1,0],3),[[1,1,0],[0,1,0],[1,0,0]])
    assert confusion_matrix_counts([],[],2).sum()==0
    with pytest.raises(ValueError): confusion_matrix_counts([0],[2],2)
