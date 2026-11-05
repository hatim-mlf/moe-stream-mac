from candidate import stable_unique_values
def test_stable_and_mixed_values():
    assert stable_unique_values([3,1,3,2,1])==[3,1,2]
    assert stable_unique_values(["b","a","b","c"])==["b","a","c"]
def test_empty():
    assert stable_unique_values([])==[]
