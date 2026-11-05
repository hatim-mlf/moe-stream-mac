import numpy as np
from candidate import pack_ragged
def test_values_offsets_and_types():
    rows=[np.array([1,2],dtype=np.int16),np.array([],dtype=np.float32),np.array([3.5])]
    flat,off=pack_ragged(rows)
    assert np.allclose(flat,[1,2,3.5]) and flat.dtype==np.float64
    assert np.array_equal(off,[0,2,2,3]) and off.dtype==np.int64
    empty,ends=pack_ragged([])
    assert empty.dtype==np.float64 and empty.shape==(0,) and np.array_equal(ends,[0])
def test_invalid():
    import pytest
    with pytest.raises(ValueError): pack_ragged([[[1]]])
    with pytest.raises(ValueError): pack_ragged([['x']])
