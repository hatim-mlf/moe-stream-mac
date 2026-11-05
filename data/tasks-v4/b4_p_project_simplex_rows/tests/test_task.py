import numpy as np
import pytest
from candidate import project_simplex_rows

def test_projection():
    x=np.array([[0.2,0.2,0.2],[-1,2,0]],float)
    y=project_simplex_rows(x)
    assert np.allclose(y,[[1/3,1/3,1/3],[0.0,1.0,0.0]])
    assert np.all(y>=0) and np.allclose(y.sum(axis=1),1)
    assert project_simplex_rows(np.empty((0,3))).shape==(0,3)
    with pytest.raises(ValueError): project_simplex_rows(np.empty((2,0)))
