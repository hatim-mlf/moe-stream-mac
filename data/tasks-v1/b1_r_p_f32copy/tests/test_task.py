import numpy as np
from candidate import owned_float32_c_array
def test_owned_and_contiguous():
    base=np.arange(24,dtype=np.float32).reshape(4,6)
    source=base[:,::2]
    got=owned_float32_c_array(source)
    np.testing.assert_array_equal(got,source)
    assert got.dtype==np.float32 and got.flags.c_contiguous
    assert not np.shares_memory(got,source)
    got[0,0]=-99
    assert source[0,0]==0
def test_dtype_conversion():
    got=owned_float32_c_array([1,2])
    assert got.dtype==np.float32
