import pytest
import numpy as np
from candidate import format_csv_record
def test_one_record_without_terminator():
    assert format_csv_record(['a','b,c','d"e'])=='a,"b,c","d""e"'
    assert format_csv_record(['line\nbreak'])=='"line\nbreak"'
    assert format_csv_record([])==''
