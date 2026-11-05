from candidate import longest_strict_rise

def test_rise():
    assert longest_strict_rise([1,2,3,1,4,5])==3
    assert longest_strict_rise([2,2,3,3])==2
    assert longest_strict_rise([])==0
    assert longest_strict_rise([-2,-1,0])==3
