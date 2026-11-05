import numpy as np
from candidate import normalize_prob_vector
def test_normalizes():
 got=normalize_prob_vector([1,3,0])
 assert got.dtype==np.float64 and np.allclose(got,[0.25,0.75,0])
def test_rejects_bad_mass():
 import pytest
 with pytest.raises(ValueError): normalize_prob_vector([0,0])
 with pytest.raises(ValueError): normalize_prob_vector([1,-1])
 with pytest.raises(ValueError): normalize_prob_vector([])
 with pytest.raises(ValueError): normalize_prob_vector([np.inf])
