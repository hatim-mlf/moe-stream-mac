import pytest
import numpy as np
from candidate import first_duplicate_item
def test_first_repeated_value():
    assert first_duplicate_item([4,2,4,2])==4
    assert first_duplicate_item([]) is None
    assert first_duplicate_item(['a','b']) is None
def test_repeated_none_is_ambiguous_value_but_supported():
    assert first_duplicate_item([None,None]) is None
