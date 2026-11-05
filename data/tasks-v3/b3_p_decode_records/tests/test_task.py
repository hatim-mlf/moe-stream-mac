import pytest
import numpy as np
from candidate import decode_records_u32be
def test_records():
    assert decode_records_u32be(b'\x00\x00\x00\x02ab\x00\x00\x00\x00')==[b'ab',b'']
    assert decode_records_u32be(b'')==[]
def test_truncation():
    with pytest.raises(ValueError): decode_records_u32be(b'\x00\x00')
    with pytest.raises(ValueError): decode_records_u32be(b'\x00\x00\x00\x03x')
    with pytest.raises(ValueError): decode_records_u32be([1])
