import numpy as np
import pytest
from candidate import project_homography2d

def test_homography():
    p=np.array([[2.,4.],[1,1]])
    h=np.array([[1,0,0],[0,1,0],[0.5,0,1]],float)
    assert np.allclose(project_homography2d(p,h),[[1,2],[2/3,2/3]])
    assert project_homography2d(np.empty((0,2)),np.eye(3)).shape==(0,2)
    with pytest.raises(ValueError): project_homography2d([[2,0]],[[1,0,0],[0,1,0],[-0.5,0,1]])
