from candidate import parse_dims
def test_parse_dims():
    assert parse_dims("3x12x1")== (3,12,1)
    assert parse_dims("7")== (7,)
def test_rejects_noncanonical():
    import pytest
    for text in ("", "0", "03x4", "2X3", "2x", "-1x2", "1 x2", "١x2"):
        with pytest.raises(ValueError): parse_dims(text)
