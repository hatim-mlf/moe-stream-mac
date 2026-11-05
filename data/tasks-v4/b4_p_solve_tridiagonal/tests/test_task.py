import numpy as np
import pytest
from candidate import solve_tridiagonal

def test_solve():
    lo=np.array([1.,1.]); d=np.array([4.,4.,4.]); up=np.array([1.,1.]); rhs=np.array([6.,6.,6.])
    got=solve_tridiagonal(lo,d,up,rhs)
    assert np.allclose(got,[1.285714285714,0.857142857143,1.285714285714])
    assert np.allclose(np.array([[4,1,0],[1,4,1],[0,1,4]])@got,rhs)
    with pytest.raises(ValueError): solve_tridiagonal([1],[1,1],[1],[2,2])
    with pytest.raises(ValueError): solve_tridiagonal([1],[1,1],[1],[1,1])
