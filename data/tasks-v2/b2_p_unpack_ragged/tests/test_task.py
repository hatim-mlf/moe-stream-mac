import numpy as np
from candidate import unpack_ragged
def test_segments_are_independent():
    flat=np.array([1,2,3],dtype=np.int16)
    rows=unpack_ragged(flat,[0,2,2,3])
    assert [r.tolist() for r in rows]==[[1,2],[],[3]]
    assert all(r.dtype==np.int16 for r in rows)
    rows[0][0]=99
    assert flat[0]==1
def test_invalid():
    import pytest
    for off in ([],[1,3],[0,2],[0,2,1,3],[0,1.5,3]):
        with pytest.raises(ValueError): unpack_ragged([1,2,3],off)
    with pytest.raises(ValueError): unpack_ragged([[1,2]],[0,2])
