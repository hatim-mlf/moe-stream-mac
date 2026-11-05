import pytest
import numpy as np
from candidate import merge_count_maps
def test_merge_and_copy():
    a={'x':2,'y':1}; got=merge_count_maps(a,{'x':3,'z':0})
    assert got=={'x':5,'y':1,'z':0} and a=={'x':2,'y':1}
    assert merge_count_maps({}, {})=={}
def test_invalid():
    for a,b in [({'x':True},{}), ({'x':-1},{}), ({1:2},{}), ([],{})]:
        with pytest.raises(ValueError): merge_count_maps(a,b)
