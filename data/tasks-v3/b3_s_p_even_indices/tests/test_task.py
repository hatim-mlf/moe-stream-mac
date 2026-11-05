import pytest
import numpy as np
from candidate import evenly_spaced_indices
def test_distribution():
    assert evenly_spaced_indices(10,4)==[0,2,5,7]
    assert evenly_spaced_indices(4,4)==[0,1,2,3]
    assert evenly_spaced_indices(8,0)==[]
def test_validation():
    for args in [(-1,0),(2,3),(True,1),(3,1.0)]:
        with pytest.raises(ValueError): evenly_spaced_indices(*args)
