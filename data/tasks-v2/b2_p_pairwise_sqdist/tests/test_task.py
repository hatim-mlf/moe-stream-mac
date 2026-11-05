import numpy as np
from candidate import pairwise_sqdist
def test_values_and_shapes():
    x=np.array([[0.,0.],[1.,2.]])
    y=np.array([[3.,4.]])
    got=pairwise_sqdist(x,y)
    assert got.dtype==np.float64 and got.shape==(2,1)
    assert np.allclose(got,[[25.],[8.]])
    assert pairwise_sqdist(np.empty((0,2)),y).shape==(0,1)
    assert np.array_equal(pairwise_sqdist(np.ones((2,0)),np.ones((3,0))),np.zeros((2,3)))
def test_invalid_and_no_mutation():
    a=np.array([[1.,2.]])
    pairwise_sqdist(a,a)
    assert np.array_equal(a,[[1.,2.]])
    import pytest
    with pytest.raises(ValueError): pairwise_sqdist([1,2],a)
    with pytest.raises(ValueError): pairwise_sqdist(a,np.ones((1,3)))
    with pytest.raises(ValueError): pairwise_sqdist([[np.inf]],[[0.]])
