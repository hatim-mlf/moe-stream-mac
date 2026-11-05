import numpy as np
from candidate import diagonal_band_mask
def test_rectangular_and_offsets():
    assert np.array_equal(diagonal_band_mask(2,4,1,0),[[False,True,False,False],[False,False,True,False]])
    assert diagonal_band_mask(0,3,0,1).shape==(0,3)
    assert np.array_equal(diagonal_band_mask(1,3,0,1),[[True,True,False]])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): diagonal_band_mask(True,2,0,0)
    with pytest.raises(ValueError): diagonal_band_mask(2,-1,0,0)
    with pytest.raises(ValueError): diagonal_band_mask(2,2,0,-1)
