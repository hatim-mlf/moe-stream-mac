import pytest
from candidate import transpose_rows
def test_transpose():
    assert transpose_rows([[1,2],[3,4]])==[[1,3],[2,4]]
    assert transpose_rows([[1],[2]])==[[1,2]]
    assert transpose_rows([])==[]
    assert transpose_rows([[],[]])==[]
def test_ragged():
    with pytest.raises(ValueError): transpose_rows([[1,2],[3]])
