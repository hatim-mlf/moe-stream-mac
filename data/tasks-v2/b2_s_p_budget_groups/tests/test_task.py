from candidate import split_by_budget
def test_greedy_groups_and_exact_fit():
    assert split_by_budget([2,3,5,1],5)==[[2,3],[5],[1]]
    assert split_by_budget([],4)==[]
    assert split_by_budget([4,0,1],5)==[[4,0,1]]
def test_invalid():
    import pytest
    with pytest.raises(ValueError): split_by_budget([1],0)
    with pytest.raises(ValueError): split_by_budget([6],5)
    with pytest.raises(ValueError): split_by_budget([True],5)
