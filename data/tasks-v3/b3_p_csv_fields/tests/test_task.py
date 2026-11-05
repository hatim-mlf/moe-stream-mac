import pytest
import numpy as np
from candidate import parse_csv_record
def test_quoted_fields():
    assert parse_csv_record('a,"b,c","d""e"')==['a','b,c','d"e']
    with pytest.raises(ValueError): parse_csv_record('')
    assert parse_csv_record('x,y\n')==['x','y']
def test_invalid():
    with pytest.raises(ValueError): parse_csv_record('"open')
    with pytest.raises(ValueError): parse_csv_record('a,b\nc,d')
