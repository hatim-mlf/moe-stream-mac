import numpy as np
import pytest
from candidate import solve_lower_triangular

def test_solve():
    a=np.array([[2.,0,0],[3,1,0],[1,-2,4]]); b=np.array([2.,5.,1.])
    x=solve_lower_triangular(a,b); np.testing.assert_allclose(a@x,b)
    assert solve_lower_triangular(np.empty((0,0)),[]).shape==(0,)
    with pytest.raises(ValueError): solve_lower_triangular([[1,2],[0,1]],[1,1])
