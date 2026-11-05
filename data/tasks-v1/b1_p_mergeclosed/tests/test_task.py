import pytest
from candidate import merge_adjacent_ranges
def test_overlap_adjacency_and_order():
    assert merge_adjacent_ranges([(5,6),(1,2),(3,4),(10,10)])==[(1,6),(10,10)]
def test_empty_and_bad():
    assert merge_adjacent_ranges([])==[]
    with pytest.raises(ValueError): merge_adjacent_ranges([(3,2)])
    with pytest.raises(ValueError): merge_adjacent_ranges([(1,2,3)])
