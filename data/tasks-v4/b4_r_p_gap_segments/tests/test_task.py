import numpy as np
from candidate import split_by_gap

def test_gap():
    got=split_by_gap([0,1,2.5,5],1.5)
    assert [x.tolist() for x in got]==[[0,1,2.5],[5]]
    assert split_by_gap([],0)==[]
    import pytest
    with pytest.raises(ValueError): split_by_gap([0,float('nan')],1)
    assert not np.shares_memory(got[0],np.asarray([0,1,2.5,5]))
