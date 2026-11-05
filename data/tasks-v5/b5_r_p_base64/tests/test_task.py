import pytest
from candidate import decode_base64_ascii

def test_decode():
    assert decode_base64_ascii('')==b''
    assert decode_base64_ascii('YWI=')==b'ab'
    for text in ('YWI','Y W I=','YWI$','YWJ='):
        with pytest.raises(ValueError): decode_base64_ascii(text)
