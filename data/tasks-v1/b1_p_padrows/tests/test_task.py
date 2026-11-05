import numpy as np
import pytest
from candidate import pad_numeric_rows
def test_padding_lengths_and_dtype():
    matrix,lengths=pad_numeric_rows([np.array([1,2],dtype=np.int16),np.array([3],dtype=np.int16)],np.int16(-1))
    np.testing.assert_array_equal(matrix,[[1,2],[3,-1]])
    np.testing.assert_array_equal(lengths,[2,1])
    assert matrix.dtype==np.int16 and lengths.dtype==np.int64
    promoted,_=pad_numeric_rows([np.array([1],dtype=np.int16),np.array([2.5],dtype=np.float32)],0.5)
    assert promoted.dtype==np.float64 and promoted[1,0]==2.5
def test_empty_and_bad():
    matrix,lengths=pad_numeric_rows([])
    assert matrix.shape==(0,0) and matrix.dtype==np.float64 and lengths.dtype==np.int64
    with pytest.raises(ValueError): pad_numeric_rows([[1],[[2]]])
