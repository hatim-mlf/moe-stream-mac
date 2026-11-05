import pytest
import numpy as np
from candidate import max_bracket_depth
def test_depth_and_nesting():
    assert max_bracket_depth('x([{}])')==3
    assert max_bracket_depth('plain')==0
    assert max_bracket_depth('()()')==1
def test_malformed():
    assert max_bracket_depth('([)]')==-1
    assert max_bracket_depth(')')==-1
    assert max_bracket_depth('{[')==-1
