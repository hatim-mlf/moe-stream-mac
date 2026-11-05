import pytest
import numpy as np
from candidate import split_on_sentinel
def test_groups_and_empty_parts():
    assert split_on_sentinel([0,1,0,0,2,3,0],0)==[[1],[2,3]]
    assert split_on_sentinel([0,0],0)==[]
    x=[1,0,2]; assert split_on_sentinel(x,0)==[[1],[2]] and x==[1,0,2]
