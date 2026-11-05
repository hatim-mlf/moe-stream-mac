import numpy as np
import pytest
from candidate import true_spans
def test_runs_and_edges():
    assert true_spans(np.array([1,1,0,1,0,1,1],dtype=bool))==[(0,2),(3,4),(5,7)]
    assert true_spans(np.array([],dtype=bool))==[]
    assert true_spans(np.array([False]))==[]
def test_requires_bool_vector():
    with pytest.raises(ValueError): true_spans([0,1])
    with pytest.raises(ValueError): true_spans(np.ones((2,2),dtype=bool))
