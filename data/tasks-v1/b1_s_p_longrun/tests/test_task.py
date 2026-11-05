from candidate import longest_equal_run
def test_runs():
    assert longest_equal_run([1,1,2,2,2,1])==3
    assert longest_equal_run(["a","b","b"])==2
    assert longest_equal_run([1,2,1,1])==2
    assert longest_equal_run([])==0
def test_generator():
    assert longest_equal_run(x for x in [4,4,4,3])==3
