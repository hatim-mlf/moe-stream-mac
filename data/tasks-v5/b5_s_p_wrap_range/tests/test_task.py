import numpy as np
import pytest
from candidate import wrap_to_range

def test_wrap():
    np.testing.assert_allclose(wrap_to_range([-181,-180,180,540],-180,180),[179,-180,-180,-180])
    with pytest.raises(ValueError): wrap_to_range([1],2,2)
    with pytest.raises(ValueError): wrap_to_range([np.inf],0,1)
