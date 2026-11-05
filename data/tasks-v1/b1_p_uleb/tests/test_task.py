import pytest
from candidate import decode_uleb32
def test_values_and_offset():
    assert decode_uleb32(bytes([0xAC,0x02]))==(300,2)
    assert decode_uleb32(bytes([9,0x80,1]),1)==(128,3)
    assert decode_uleb32(bytes([0xff,0xff,0xff,0xff,0x0f]))==(0xffffffff,5)
def test_bad_streams():
    for data in (b"",b"\x80",b"\xff\xff\xff\xff\x10",b"\x80\x80\x80\x80\x80"):
        with pytest.raises(ValueError): decode_uleb32(data)
    with pytest.raises(ValueError): decode_uleb32(b"\0",2)
