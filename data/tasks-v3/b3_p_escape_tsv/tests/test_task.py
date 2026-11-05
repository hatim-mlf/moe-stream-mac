import pytest
import numpy as np
from candidate import escape_tsv_cell
def test_escapes():
    assert escape_tsv_cell('a\\b\tc\rd\ne')=='a\\\\b\\tc\\rd\\ne'
    assert escape_tsv_cell('plain')=='plain'
def test_backslash_before_escape():
    assert escape_tsv_cell('\\n')=='\\\\n'
