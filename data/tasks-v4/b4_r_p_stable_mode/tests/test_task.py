import pytest
from candidate import stable_mode

def test_mode():
    assert stable_mode(['a','z','z','a'])=='a'
    assert stable_mode([('x',1),('y',2),('y',2)])==('y',2)
    with pytest.raises(ValueError): stable_mode([])
