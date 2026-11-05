import numpy as np
import pytest
from candidate import derivative_nonuniform

def test_derivative():
    x=np.array([0.,1.,3.,6.]); y=x*x
    assert np.allclose(derivative_nonuniform(y,x),[1,2,6,9])
    assert derivative_nonuniform([4,9],[2,5]).tolist()==[5/3,5/3]
    with pytest.raises(ValueError): derivative_nonuniform([1,2],[1,1])
