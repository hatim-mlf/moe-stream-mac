import pytest
import numpy as np
from candidate import pack_i16le
def test_encoding():
    assert pack_i16le(np.array([-2,0,0x1234],dtype=np.int32))==b"\xfe\xff\x00\x00\x34\x12"
    assert pack_i16le(np.array([],dtype=np.int16))==b""
def test_invalid():
    with pytest.raises(ValueError): pack_i16le([1.5])
    with pytest.raises(ValueError): pack_i16le([32768])
    with pytest.raises(ValueError): pack_i16le(np.array([True]))
