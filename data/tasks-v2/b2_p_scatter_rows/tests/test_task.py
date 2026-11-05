import numpy as np
from candidate import scatter_add_rows
def test_duplicates_accumulate_and_inputs_stay_unchanged():
    base=np.zeros((3,2),dtype=np.int16); idx=np.array([1,1,2]); updates=np.array([[1,2],[3,4],[5,6]])
    got=scatter_add_rows(base,idx,updates)
    assert got.dtype==np.float64
    assert np.array_equal(got,[[0,0],[4,6],[5,6]])
    assert np.array_equal(base,np.zeros((3,2)))
def test_empty_and_validation():
    import pytest
    assert np.array_equal(scatter_add_rows(np.ones((2,1)),[],np.empty((0,1))),[[1],[1]])
    with pytest.raises(ValueError): scatter_add_rows(np.zeros((2,1)),[2],[[1]])
    with pytest.raises(ValueError): scatter_add_rows(np.zeros((2,1)),[0],[[1,2]])
    with pytest.raises(ValueError): scatter_add_rows(np.zeros((2,1)),[0.5],[[1]])
