from candidate import parse_size_bytes
def test_units_and_zero():
    assert parse_size_bytes("0B")==0
    assert parse_size_bytes("12B")==12
    assert parse_size_bytes("1KiB")==1024
    assert parse_size_bytes("2MiB")==2*1024**2
    assert parse_size_bytes("3GiB")==3*1024**3
def test_rejects_noncanonical_forms():
    import pytest
    for text in ("", "01B", "-1B", "+1B", "1kb", "1 B", "1KiB ", "B"):
        with pytest.raises(ValueError): parse_size_bytes(text)
