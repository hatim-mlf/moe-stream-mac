import numpy as np
import pytest
from candidate import lower_bound_index

def test_lower_bound():
    assert lower_bound_index([1,2,2,5],2)==1
    assert lower_bound_index([1,2],4)==2
    assert lower_bound_index([],3)==0
    with pytest.raises(ValueError): lower_bound_index([2,1],1)
