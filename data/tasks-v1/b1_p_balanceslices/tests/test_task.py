import pytest
from candidate import balanced_slices
def test_even_and_remainder():
    assert balanced_slices(8,4)==[(0,2),(2,4),(4,6),(6,8)]
    assert balanced_slices(8,3)==[(0,3),(3,6),(6,8)]
def test_more_parts_and_empty():
    assert balanced_slices(2,4)==[(0,1),(1,2),(2,2),(2,2)]
    assert balanced_slices(0,2)==[(0,0),(0,0)]
def test_invalid():
    with pytest.raises(ValueError): balanced_slices(-1,2)
    with pytest.raises(ValueError): balanced_slices(2,0)
