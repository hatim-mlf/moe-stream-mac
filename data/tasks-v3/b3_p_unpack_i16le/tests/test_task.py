import pytest
import numpy as np
from candidate import unpack_i16le
def test_signed_values_and_copy():
    raw=bytearray(b"\xff\xff\x34\x12")
    got=unpack_i16le(raw)
    assert got.dtype==np.int16 and np.array_equal(got,[-1,0x1234])
    raw[0]=0; assert got[0]==-1
    assert unpack_i16le(b"").shape==(0,)
def test_invalid():
    with pytest.raises(ValueError): unpack_i16le(b"x")
    with pytest.raises(ValueError): unpack_i16le([1,2])
