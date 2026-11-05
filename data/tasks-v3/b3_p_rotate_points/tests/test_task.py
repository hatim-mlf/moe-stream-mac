import pytest
import numpy as np
from candidate import rotate_points2d
def test_rotation_and_shape():
    p=np.array([[1.,0.],[0,1.]])
    assert np.allclose(rotate_points2d(p,np.pi/2),[[0,1],[-1,0]],atol=1e-12)
    assert rotate_points2d(np.empty((2,0,2)),0).shape==(2,0,2)
    assert p[0,0]==1
def test_invalid():
    with pytest.raises(ValueError): rotate_points2d([1,2,3],0)
    with pytest.raises(ValueError): rotate_points2d([[np.inf,0]],0)
