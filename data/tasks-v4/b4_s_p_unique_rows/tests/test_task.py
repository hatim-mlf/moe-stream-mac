import numpy as np
import pytest
from candidate import first_unique_rows

def test_unique_rows():
    x=np.array([[2,1],[1,2],[2,1],[3,3]])
    assert first_unique_rows(x).tolist()==[0,1,3]
    assert first_unique_rows(np.empty((0,2))).dtype==np.int64
    with pytest.raises(ValueError): first_unique_rows([[1],[float('nan')]])
