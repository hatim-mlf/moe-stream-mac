import pytest
from candidate import stable_topological_order

def test_order():
    assert stable_topological_order(['a','b','c','d'], [('a','d')]) == ['a','b','c','d']
    assert stable_topological_order(['x','y','z'], [('z','x'),('x','y')]) == ['z','x','y']
    with pytest.raises(ValueError): stable_topological_order([1,1], [])
    with pytest.raises(ValueError): stable_topological_order([1], [(1,2)])
    with pytest.raises(ValueError): stable_topological_order([1,2], [(1,2),(2,1)])
