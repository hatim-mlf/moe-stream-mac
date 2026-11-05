import pytest
import numpy as np
from candidate import row_entropy

def test_entropy_normalizes_rows_and_handles_zero_terms():
    p=np.array([[1.,3.],[1,0],[1,1]])
    got=row_entropy(p)
    assert np.allclose(got,[-(.25*np.log(.25)+.75*np.log(.75)),0,np.log(2)])
    assert p[0,0]==1
    assert np.allclose(row_entropy([[1e308,1e308]]),[np.log(2)])
def test_empty_and_invalid():
    assert row_entropy(np.empty((0,3))).shape==(0,)
    for p in ([[0,0]],[[-1,2]],[[np.inf,1]],[[0,0]],[1,2]):
        with pytest.raises(ValueError): row_entropy(p)
