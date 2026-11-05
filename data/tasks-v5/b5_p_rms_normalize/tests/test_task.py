import numpy as np
import pytest
from candidate import normalize_by_rms

def test_rms():
    x=np.array([[1.,-1.],[3.,3.]])
    np.testing.assert_allclose(normalize_by_rms(x,axis=1),[[1,-1],[1,1]])
    np.testing.assert_array_equal(normalize_by_rms([[0.,0.]]),[[0,0]])
    with pytest.raises(ValueError): normalize_by_rms(x,axis=True)
